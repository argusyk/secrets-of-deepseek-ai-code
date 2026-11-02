"""Lab 2. From MHA to MLA: the KV cache and matrix absorption (Chapters 2, 4).

Steps:
  1. check that the absorbed MLA (formulas (4.5)-(4.6)) gives the same outputs
     as the naive one (float64 and float32);
  2. KV-cache size per token: MHA, GQA, MQA and MLA for TinySeek and for the
     DeepSeek-V3 configuration (formulas (4.1), (4.10)); compare the bytes
     really held by the cache tensors with the formula;
  3. time one decoding step with a long cache: naive vs absorbed MLA;
  4. train MHA and MLA variants of TinySeek for the same number of steps and
     compare validation loss.

Run (from code/):
    python tinyseek_labs/lab2_mla.py
    python tinyseek_labs/lab2_mla.py --kv-lora-rank 16 --rope-dim 8 --steps 300
Outputs: runs/lab2/kv_cache.csv, training.csv, loss.svg
"""
import argparse
import time
from pathlib import Path

from _common import RUNS, Timer, print_table, run_dir, smooth, svg_plot, write_csv
import torch

from tinyseek.config import TinySeekConfig, get_preset
from tinyseek.data import PackedDataset, prepare_data
from tinyseek.model import (MultiHeadLatentAttention, TinySeekModel, cache_tensor_bytes,
                            count_parameters, kv_cache_bytes)
from tinyseek.train import evaluate_loss, train

V3_JSON = Path(__file__).resolve().parents[1] / "tinyseek" / "configs" / "deepseek_v3.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kv-lora-rank", type=int, default=32, help="d_c")
    ap.add_argument("--rope-dim", type=int, default=16, help="d_h^R")
    ap.add_argument("--q-lora-rank", type=int, default=48, help="d_c' (0 = off)")
    ap.add_argument("--steps", type=int, default=400, help="training steps per model")
    ap.add_argument("--context", type=int, default=256, help="context for timing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(2), Timer()
    torch.manual_seed(args.seed)
    mla_cfg = get_preset("tiny_mla", kv_lora_rank=args.kv_lora_rank,
                         qk_rope_head_dim=args.rope_dim, q_lora_rank=args.q_lora_rank)

    # 1. absorption is exact -------------------------------------------------
    attn = MultiHeadLatentAttention(mla_cfg)
    rows = []
    for dtype in (torch.float64, torch.float32):
        a = MultiHeadLatentAttention(mla_cfg).to(dtype)
        a.load_state_dict(attn.state_dict())
        x = torch.randn(2, 64, mla_cfg.hidden_size, dtype=dtype)
        pos = torch.arange(64)
        with torch.no_grad():
            d = (a(x, pos, absorbed=False) - a(x, pos, absorbed=True)).abs().max().item()
        rows.append({"dtype": str(dtype).split(".")[1], "max_abs_diff": d})
    print_table(rows, "Naive vs absorbed MLA (same weights)")

    # 2. cache size ----------------------------------------------------------
    base = get_preset("tiny_dense")
    variants = {"MHA": base, "GQA (2 KV heads)": base.replace(num_key_value_heads=2),
                "MQA (1 KV head)": base.replace(num_key_value_heads=1), "MLA": mla_cfg}
    T = 256
    krow = []
    for name, c in variants.items():
        m = TinySeekModel(c).eval()
        cache = m.new_cache()
        with torch.no_grad():
            m(torch.randint(0, c.vocab_size, (1, T)), cache=cache)
        per_tok = kv_cache_bytes(c, 1, bytes_per_elem=1) // c.num_hidden_layers
        krow.append({"model": name, "elems/token/layer": per_tok,
                     "formula_KB_fp32": kv_cache_bytes(c, T, bytes_per_elem=4) / 1024,
                     "measured_KB": cache_tensor_bytes(cache) / 1024,
                     "vs_MHA": round(kv_cache_bytes(base, T) / kv_cache_bytes(c, T), 2)})
    v3 = TinySeekConfig.from_json(V3_JSON)
    v3_mha = v3.replace(attention="mha")
    for name, c in (("V3 as MHA (128 heads)", v3_mha), ("V3 with MLA", v3)):
        b = kv_cache_bytes(c, 128 * 1024, bytes_per_elem=2)
        krow.append({"model": name, "elems/token/layer": kv_cache_bytes(c, 1, bytes_per_elem=1)
                     // c.num_hidden_layers, "formula_KB_fp32": float("nan"),
                     "measured_KB": float("nan"), "vs_MHA": round(
                         kv_cache_bytes(v3_mha, 1) / kv_cache_bytes(c, 1), 1),
                     "GB_at_128K_bf16": round(b / 1e9, 1)})
    print_table(krow, f"KV cache: TinySeek at T={T} (fp32 tensors), DeepSeek-V3 at 128K (BF16)")
    write_csv(out / "kv_cache.csv", krow)

    # 3. decode-step timing -------------------------------------------------
    m = TinySeekModel(mla_cfg).eval()
    trow = []
    for absorbed in (False, True):
        m.set_absorbed(absorbed)
        cache = m.new_cache()
        with torch.no_grad():
            m(torch.randint(0, 2048, (8, args.context - 32)), cache=cache)
            t0 = time.perf_counter()
            for _ in range(32):
                m(torch.randint(0, 2048, (8, 1)), cache=cache)
        trow.append({"form": "absorbed" if absorbed else "naive",
                     "ms_per_step(batch 8)": (time.perf_counter() - t0) / 32 * 1e3})
    print_table(trow, f"One decoding step with a cache of ~{args.context} tokens")

    # 4. MHA vs MLA training --------------------------------------------------
    tok, data = prepare_data(RUNS / "data", 2048, args.seed)
    tr, va = PackedDataset(data["train"], 256), PackedDataset(data["val"], 256)
    res, curves = [], {}
    for name, c in (("tiny_dense (MHA)", base), ("tiny_mla (MLA)", mla_cfg)):
        model, hist = train(c, tr, args.steps, seed=args.seed, verbose=False, log_every=10)
        ev = evaluate_loss(model, va)
        res.append({"model": name, "params": count_parameters(model),
                    "cache_elems/token": kv_cache_bytes(c, 1, bytes_per_elem=1),
                    "train_loss_last": round(sum(h["loss"] for h in hist[-5:]) / 5, 4),
                    "val_loss": round(ev["val_loss"], 4), "val_ppl": round(ev["val_ppl"], 2),
                    "seconds": hist[-1]["time"]})
        curves[name] = ([h["step"] for h in hist], smooth([h["loss"] for h in hist]))
    print_table(res, f"Training {args.steps} steps on the local corpus")
    write_csv(out / "training.csv", res)
    svg_plot(out / "loss.svg", curves, "MHA vs MLA: training loss", "step", "loss")
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
