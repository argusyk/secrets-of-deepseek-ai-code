"""Lab 4. MTP and training TinySeek-V3 with the V3 schedule (Chapters 7-9).

Steps:
  1. assemble tiny_v3 and count its parameters with the formulas of Chapter 8;
  2. plot the V3-shaped learning-rate schedule (warm-up, constant, cosine,
     two constant tails; Section 9.4);
  3. pre-train tiny_v3 with an MTP module (lambda = 0.3, formula (7.6)) and the
     same model without MTP for the same number of steps; compare validation
     loss and the quality of the MTP head (how often its top-1 guess of token
     t+2 is right and agrees with the main model - a preview of the
     acceptance rate of speculative decoding, Section 7.5);
  4. a mini scaling experiment: three widths, a power-law fit L = a * N^(-b).

The MTP model is saved to runs/lab4/v3 and reused by labs 6-8.

Run (from code/):
    python tinyseek_labs/lab4_train.py
    python tinyseek_labs/lab4_train.py --steps 1500 --mtp-lambda 0.1
Outputs: runs/lab4/summary.csv, loss.svg, lr.svg, scaling.csv
"""
import argparse
import math

import numpy as np
from _common import RUNS, Timer, print_table, run_dir, smooth, svg_plot, write_csv
import torch

from tinyseek.config import get_preset
from tinyseek.data import PackedDataset, prepare_data
from tinyseek.model import count_parameters
from tinyseek.train import evaluate_loss, lr_schedule, save_checkpoint, train


@torch.no_grad()
def mtp_quality(model, ds: PackedDataset, batches: int = 4) -> dict:
    """Top-1 accuracy of the MTP head on t+2 and its agreement with the main model."""
    hit = agree = n = 0
    for x, y in ds.iter_batches(8, batches):
        h = model.hidden_states(x)
        main_next = model.logits_from_hidden(h).argmax(-1)           # guesses of t+1
        lg, _ = model.mtp_logits(0, h[:, :-1], x[:, 1:])             # guesses of t+2
        mtp = lg.argmax(-1)
        hit += int((mtp == y[:, 1:]).sum())
        agree += int((mtp == main_next[:, 1:]).sum())
        n += mtp.numel()
    return {"mtp_top1_acc": hit / n, "mtp_agrees_with_main": agree / n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mtp-lambda", type=float, default=0.3)
    ap.add_argument("--scaling-steps", type=int, default=150, help="0 = skip step 4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(4), Timer()
    cfg = get_preset("tiny_v3", mtp_lambda=args.mtp_lambda)

    # 1-2. parameters and schedule ------------------------------------------------
    print(f"tiny_v3: total {cfg.num_parameters():,}, active {cfg.num_parameters(True):,}, "
          f"MTP module {cfg.num_parameters(include_mtp=True) - cfg.num_parameters():,}")
    xs = list(range(args.steps))
    svg_plot(out / "lr.svg", {"lr": (xs, [lr_schedule(s, args.steps, args.lr, args.warmup)
                                          for s in xs])},
             "V3-shaped learning-rate schedule", "step", "learning rate")

    # 3. MTP vs no MTP ---------------------------------------------------------------
    tok, data = prepare_data(RUNS / "data", cfg.vocab_size, args.seed)
    tr, va = PackedDataset(data["train"], 256), PackedDataset(data["val"], 256)
    rows, curves = [], {}
    for name, c in (("with MTP", cfg), ("no MTP", cfg.replace(num_nextn_predict_layers=0))):
        model, hist = train(c, tr, args.steps, args.batch_size, args.lr, args.warmup,
                            seed=args.seed, val_dataset=va, eval_every=max(50, args.steps // 6),
                            log_path=out / f"log_{name.replace(' ', '_')}.jsonl",
                            verbose=False)
        ev = evaluate_loss(model, va, max_batches=16)
        row = {"model": name, "params_main": count_parameters(model),
               "val_loss": round(ev["val_loss"], 4), "val_ppl": round(ev["val_ppl"], 2),
               "val_mtp_loss": round(ev["val_mtp"], 4) if c.num_nextn_predict_layers else "-",
               "maxvio_end": round(hist[-1]["maxvio"], 3), "seconds": hist[-1]["time"]}
        if c.num_nextn_predict_layers:
            row.update({k: round(v, 3) for k, v in mtp_quality(model, va).items()})
            save_checkpoint(model, out / "v3", RUNS / "data" / f"tokenizer_{cfg.vocab_size}.json")
            curves["MTP loss (t+2)"] = ([h["step"] for h in hist], smooth([h["mtp"] for h in hist]))
        curves[f"LM loss, {name}"] = ([h["step"] for h in hist], smooth([h["loss"] for h in hist]))
        val = [h for h in hist if "val_loss" in h]
        curves[f"val loss, {name}"] = ([h["step"] for h in val], [h["val_loss"] for h in val])
        rows.append(row)
    print_table(rows, f"tiny_v3 after {args.steps} steps (batch {args.batch_size} x 256 tokens)")
    write_csv(out / "summary.csv", rows)
    svg_plot(out / "loss.svg", curves, "Training with and without MTP", "step", "loss")

    # 4. mini scaling law ------------------------------------------------------------
    if args.scaling_steps:
        srows = []
        for d in (32, 64, 128):
            c = get_preset("tiny_dense", hidden_size=d, intermediate_size=2 * d,
                           num_attention_heads=max(1, d // 32))
            m, _ = train(c, tr, args.scaling_steps, args.batch_size, args.lr, 20,
                         seed=args.seed, verbose=False)
            n_nonemb = count_parameters(m) - 2 * c.vocab_size * d
            srows.append({"d": d, "non_embedding_params": n_nonemb,
                          "val_loss": round(evaluate_loss(m, va)["val_loss"], 4)})
        x = np.log([r["non_embedding_params"] for r in srows])
        y = np.log([r["val_loss"] for r in srows])
        b, a = np.polyfit(x, y, 1)
        print_table(srows, f"Scaling: {args.scaling_steps} steps per model")
        print(f"fit: L = {math.exp(a):.2f} * N^({b:.3f})")
        if b >= 0:
            print("the loss does not fall with N yet: the runs are too short for the larger "
                  "models to use their capacity (try --scaling-steps 600)")
        write_csv(out / "scaling.csv", srows)
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
