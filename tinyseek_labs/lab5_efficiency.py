"""Lab 5. Training efficiency: FP8 emulation and pipeline parallelism (Chapters 10-11).

Steps:
  1. quantization error of E4M3 with one scale per tensor vs 1x128 tiles and
     128x128 blocks (formulas (10.3)-(10.5)) on real TinySeek activations;
  2. train tiny_v3 in FP32 and with emulated FP8 GEMMs from the same seed and
     compare the loss curves (V3 reports a relative gap below 0.25%);
  3. measure forward/backward/weight-gradient times of a TinySeek block on this
     CPU, split the model into p "virtual accelerators" and simulate 1F1B and
     DualPipe (formula (11.4)), with and without all-to-all communication;
  4. count all-to-all traffic for real routing decisions of TinySeek when the
     experts are spread over EP devices, with and without deduplication.

Run (from code/):
    python tinyseek_labs/lab5_efficiency.py
    python tinyseek_labs/lab5_efficiency.py --block 32 --p 8 --m 32 --comm 0.5
Outputs: runs/lab5/fp8.csv, fp8_loss.svg, pipeline.csv, pipeline.txt, alltoall.csv
"""
import argparse

from _common import RUNS, Timer, base_model, print_table, run_dir, smooth, svg_plot, write_csv
import torch

from tinyseek.config import get_preset
from tinyseek.data import PackedDataset, prepare_data
from tinyseek.fp8 import convert_to_fp8, fake_quant_e4m3, relative_error
from tinyseek.model import TinySeekModel
from tinyseek.parallel_sim import (all_to_all_bytes, ascii_timeline, bubble_formula,
                                   compare_schedules, measure_stage_times)
from tinyseek.train import evaluate_loss, set_seed, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=300, help="FP32/FP8 training steps")
    ap.add_argument("--block", type=int, default=128, help="tile width for FP8 scaling")
    ap.add_argument("--p", type=int, default=4, help="pipeline stages (even)")
    ap.add_argument("--m", type=int, default=16, help="micro-batches")
    ap.add_argument("--comm", type=float, default=0.5, help="all-to-all time per chunk / F")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(5), Timer()

    # 1. quantization error on real activations ----------------------------------
    model, tok, _ = base_model()
    _, data = prepare_data(RUNS / "data", 2048, args.seed, verbose=False)
    x = torch.as_tensor(data["val"][:1024].astype("int64"))[None]
    acts = {}
    hooks = [model.layers[i].ffn_norm.register_forward_hook(
        lambda m, i, o, k=i: acts.__setitem__(f"layer{k}", o.detach()[0])) for i in range(4)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    qrows = []
    for name, a in acts.items():
        peak = float(a.abs().max() / a.abs().mean())
        qrows.append({"tensor": f"{name} FFN input", "max/mean": round(peak, 1),
                      "per_tensor": relative_error(fake_quant_e4m3(a, None), a),
                      f"tile_1x{args.block}": relative_error(fake_quant_e4m3(a, (1, args.block)), a)})
    w = model.layers[1].attn.wkv_b.weight.detach()
    qrows.append({"tensor": "weight wkv_b", "max/mean": round(float(w.abs().max() / w.abs().mean()), 1),
                  "per_tensor": relative_error(fake_quant_e4m3(w, None), w),
                  f"tile_1x{args.block}": relative_error(
                      fake_quant_e4m3(w, (args.block, args.block)), w)})
    print_table(qrows, "E4M3 relative error: one scale per tensor vs fine-grained scales")

    # 2. FP32 vs FP8 training ---------------------------------------------------------
    cfg = get_preset("tiny_v3")
    tr, va = PackedDataset(data["train"], 256), PackedDataset(data["val"], 256)
    rows, curves, losses = [], {}, {}
    for label in ("fp32", "fp8"):
        set_seed(args.seed)
        m = TinySeekModel(cfg)
        if label == "fp8":
            convert_to_fp8(m, (1, args.block), (args.block, args.block))
        m, hist = train(cfg, tr, args.steps, seed=args.seed, model=m, verbose=False, log_every=5)
        ev = evaluate_loss(m, va)
        losses[label] = [h["loss"] for h in hist]
        rows.append({"precision": label, "val_loss": round(ev["val_loss"], 4),
                     "seconds": hist[-1]["time"]})
        curves[label] = ([h["step"] for h in hist], smooth(losses[label], 8))
    tail = len(losses["fp32"]) // 4
    gap = (sum(losses["fp8"][-tail:]) - sum(losses["fp32"][-tail:])) / sum(losses["fp32"][-tail:])
    for r in rows:
        r["train_loss_gap_vs_fp32"] = f"{gap:+.2%}" if r["precision"] == "fp8" else "0"
    print_table(rows, f"FP32 vs emulated FP8 (tiles 1x{args.block}, blocks "
                      f"{args.block}x{args.block}), {args.steps} steps")
    write_csv(out / "fp8.csv", qrows + rows)
    svg_plot(out / "fp8_loss.svg", curves, "FP32 vs emulated FP8", "step", "training loss")

    # 3. pipeline schedules -----------------------------------------------------------
    t = measure_stage_times(cfg, p=args.p)
    F, B, W = 1.0, t["B"] / t["F"], t["W"] / t["F"]
    print(f"\nmeasured per stage (p={args.p}): F={t['F'] * 1e3:.1f} ms, B={t['B'] * 1e3:.1f} ms, "
          f"W={t['W'] * 1e3:.1f} ms  ->  B/F={B:.2f}, W/F={W:.2f}")
    prow, text = [], []
    for p in sorted({args.p, 2 * args.p}):
        for comm in (0.0, args.comm):
            res = compare_schedules(p, max(args.m, 2 * p), F, B, W, comm)
            for name, r in res.items():
                prow.append({"p": p, "m": max(args.m, 2 * p), "comm/F": comm, "schedule": name,
                             "makespan": round(r["makespan"], 2),
                             "bubble": round(r["bubble_time"], 2),
                             "bubble_share": round(r["bubble_share"], 3),
                             "formula": round(bubble_formula(name, p, F, B, W), 2)})
            if p == args.p and comm == 0.0:
                for name, r in res.items():
                    text.append(f"{name}, p={p}, m={max(args.m, 2 * p)}\n" +
                                ascii_timeline(r["timeline"], r["makespan"], unit=F / 2))
    print_table(prow, "1F1B vs DualPipe on virtual accelerators (times in units of F)")
    write_csv(out / "pipeline.csv", prow)
    (out / "pipeline.txt").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print("timelines: F/f forward, B/b backward, W/w weight grad, X overlapped pair, . bubble"
          f" -> {out / 'pipeline.txt'}")

    # 4. all-to-all traffic ----------------------------------------------------------
    with torch.no_grad():
        model(x)
    moe = model.moe_layers()[0]
    arows = []
    for ep in (2, 4, 8):
        r = all_to_all_bytes(moe.last_topk, moe.n_routed, ep, cfg.hidden_size)
        arows.append({"EP devices": ep, "copies_naive": r["copies_naive"],
                      "copies_dedup": r["copies_dedup"],
                      "KB_naive": round(r["bytes_naive"] / 1024, 1),
                      "KB_dedup": round(r["bytes_dedup"] / 1024, 1),
                      "bytes/token": round(r["bytes_per_token_dedup"], 1)})
    print_table(arows, f"All-to-all of one MoE layer for {x.shape[1]} tokens "
                       "(FP8 dispatch + BF16 combine)")
    write_csv(out / "alltoall.csv", arows)
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
