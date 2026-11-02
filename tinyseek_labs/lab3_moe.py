"""Lab 3. Mixture of experts and auxiliary-loss-free balancing (Chapters 5-6).

Steps:
  1. count total and active parameters of the dense and MoE variants
     (formula (5.5)) and the number of expert combinations C(N_r, K_r);
  2. train the MoE model three times on the same data: without balancing,
     with the auxiliary loss (6.6)-(6.7) and with the V3 bias rule (6.3)-(6.5);
     log MaxVio (6.9) at every step;
  3. look at expert specialization: which experts the tokens of code, UK, EN
     and arithmetic text go to.

Run (from code/):
    python tinyseek_labs/lab3_moe.py
    python tinyseek_labs/lab3_moe.py --gamma 0.003 --alpha 0.03 --experts 16 --top-k 4
Outputs: runs/lab3/balancing.csv, maxvio.svg, experts_by_source.csv
"""
import argparse
from math import comb

from _common import RUNS, Timer, print_table, run_dir, smooth, svg_plot, write_csv
import torch

from tinyseek.config import get_preset
from tinyseek.data import PackedDataset, prepare_data
from tinyseek.model import TinySeekModel, count_parameters, max_violation
from tinyseek.train import evaluate_loss, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--experts", type=int, default=8, help="N_r routed experts")
    ap.add_argument("--top-k", type=int, default=2, help="K_r")
    ap.add_argument("--shared", type=int, default=1, help="N_s shared experts")
    ap.add_argument("--gamma", type=float, default=1e-3, help="bias update speed")
    ap.add_argument("--alpha", type=float, default=1e-2, help="aux loss weight")
    ap.add_argument("--scoring", default="sigmoid", choices=["sigmoid", "softmax"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(3), Timer()
    width = 128 // args.top_k                 # keep the active routed width fixed
    moe_cfg = get_preset("tiny_moe", n_routed_experts=args.experts,
                         num_experts_per_tok=args.top_k, n_shared_experts=args.shared,
                         moe_intermediate_size=max(8, width), bias_update_speed=args.gamma,
                         aux_loss_alpha=args.alpha, scoring_func=args.scoring)

    # 1. parameters ------------------------------------------------------------
    rows = []
    for name, c in (("tiny_dense", get_preset("tiny_dense")), ("tiny_moe", moe_cfg)):
        m = TinySeekModel(c)
        rows.append({"model": name, "total": count_parameters(m),
                     "active": count_parameters(m, active=True),
                     "combinations": comb(c.n_routed_experts, c.num_experts_per_tok)
                     if c.n_routed_experts else 1})
    print_table(rows, "Parameters (formula (5.5)); combinations C(N_r, K_r) per token")

    # 2. three balancing strategies -------------------------------------------
    tok, data = prepare_data(RUNS / "data", 2048, args.seed)
    tr, va = PackedDataset(data["train"], 256), PackedDataset(data["val"], 256)
    res, curves, models = [], {}, {}
    for bal in ("none", "aux", "bias"):
        c = moe_cfg.replace(balance=bal)
        model, hist = train(c, tr, args.steps, seed=args.seed, verbose=False, log_every=5)
        ev = evaluate_loss(model, va)
        tail = hist[len(hist) * 3 // 4:]
        res.append({"balance": bal, "maxvio_last25%": round(sum(h["maxvio"] for h in tail)
                                                            / len(tail), 3),
                    "maxvio_peak": round(max(h["maxvio"] for h in hist), 3),
                    "val_loss": round(ev["val_loss"], 4), "val_ppl": round(ev["val_ppl"], 2),
                    "seconds": hist[-1]["time"]})
        curves[bal] = ([h["step"] for h in hist], smooth([h["maxvio"] for h in hist], 8))
        models[bal] = model
    print_table(res, f"Balancing strategies after {args.steps} steps "
                     f"(gamma={args.gamma}, alpha={args.alpha})")
    write_csv(out / "balancing.csv", res)
    svg_plot(out / "maxvio.svg", curves, "MaxVio during training (moving average)",
             "step", "MaxVio")

    # 3. specialization -------------------------------------------------------
    model = models["bias"].eval()
    erows = []
    for src, arr in data["val_by_source"].items():
        if len(arr) < 600:
            continue
        x = torch.as_tensor(arr[:2048].astype("int64"))[None]
        with torch.no_grad():
            model(x)
        moe = model.moe_layers()[-1]
        load = moe.last_load / moe.last_load.sum()
        row = {"source": src, "maxvio": round(max_violation(moe.last_load), 3)}
        row.update({f"e{i}": round(float(v), 3) for i, v in enumerate(load)})
        erows.append(row)
    print_table(erows, "Share of routed tokens per expert in the last MoE layer (bias model)")
    write_csv(out / "experts_by_source.csv", erows)
    print(f"final biases of the last MoE layer: "
          f"{[round(float(b), 4) for b in model.moe_layers()[-1].bias]}")
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
