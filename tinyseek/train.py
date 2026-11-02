"""Pre-training loop with the DeepSeek-V3 learning-rate schedule (Chapters 6, 9).

`train` runs AdamW (beta = (0.9, 0.95), weight decay 0.1, gradient clipping
1.0 as in the V3 report) on a `PackedDataset`, updates the MoE balancing
biases after every step (formula (6.5)) and writes a JSONL log with the
loss, perplexity, MTP loss and MaxVio (6.9).

`lr_schedule` reproduces the shape of the V3 schedule (Section 9.4) in
fractions of the run: linear warm-up -> constant peak until 67.6% of the
steps (10T of 14.8T tokens) -> cosine decay to 10% of the peak by 96.6%
(14.3T) -> constant 10% -> 3.3% of the peak for the last 1.1% (167B tokens).

CLI::

    python -m tinyseek.train --preset tiny_v3 --steps 1500 --out runs/v3
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .config import TinySeekConfig, get_preset, presets
from .data import DEFAULT_DATA_DIR, PackedDataset, prepare_data
from .model import TinySeekModel, max_violation
from .tokenizer import ByteBPETokenizer


def set_seed(seed: int) -> torch.Generator:
    """Seed Python, NumPy and PyTorch; return a torch Generator for batches."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def lr_schedule(step: int, total_steps: int, peak_lr: float, warmup: int = 100) -> float:
    """Learning rate at ``step`` following the V3 schedule (Section 9.4)."""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    f = step / max(1, total_steps)
    const_end, cos_end, mid_end = 10.0 / 14.8, 14.3 / 14.8, 14.633 / 14.8
    lo = 0.1 * peak_lr                       # 2.2e-5 / 2.2e-4
    if f < const_end:
        return peak_lr
    if f < cos_end:
        p = (f - const_end) / (cos_end - const_end)
        return lo + 0.5 * (peak_lr - lo) * (1 + math.cos(math.pi * p))
    if f < mid_end:
        return lo
    return peak_lr * 7.3e-6 / 2.2e-4         # final 7.3e-6 / 2.2e-4


def moe_max_violation(model: TinySeekModel) -> Optional[float]:
    """Mean MaxVio over the MoE layers of the main model for the last batch."""
    vals = [max_violation(m.last_load) for name, m in model.named_modules()
            if type(m).__name__ == "MoE" and not name.startswith("mtp.")]
    return float(np.mean(vals)) if vals else None


@torch.no_grad()
def evaluate_loss(model: TinySeekModel, dataset: PackedDataset, batch_size: int = 8,
                  max_batches: int = 8) -> dict:
    """Mean LM and MTP losses on a dataset (deterministic pass)."""
    was = model.training
    model.eval()
    lm, mtp, n = 0.0, 0.0, 0
    for x, y in dataset.iter_batches(batch_size, max_batches):
        _, losses = model(x, y)
        lm += float(losses["lm"])
        mtp += float(losses["mtp"])
        n += 1
    model.train(was)
    return {"val_loss": lm / n, "val_ppl": math.exp(lm / n), "val_mtp": mtp / n}


def train(config: TinySeekConfig, dataset: PackedDataset, steps: int,
          batch_size: int = 8, lr: float = 3e-3, warmup: int = 100,
          weight_decay: float = 0.1, grad_clip: float = 1.0, seed: int = 0,
          model: Optional[TinySeekModel] = None, val_dataset: Optional[PackedDataset] = None,
          eval_every: int = 250, log_path: Optional[str | Path] = None,
          log_every: int = 10, callback: Optional[Callable[[int, dict], None]] = None,
          verbose: bool = True) -> tuple[TinySeekModel, list[dict]]:
    """Train ``model`` (or a fresh one built from ``config``) for ``steps`` steps.

    Returns the model and the list of logged records. Each record contains
    ``step, lr, loss, ppl, mtp, aux, maxvio, time`` (and ``val_*`` every
    ``eval_every`` steps).
    """
    gen = set_seed(seed)
    if model is None:
        model = TinySeekModel(config)
    model.train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), eps=1e-8)
    log_f = open(log_path, "w", encoding="utf-8") if log_path else None
    history: list[dict] = []
    moes = model.moe_layers()
    t0 = time.time()
    for step in range(steps):
        cur_lr = lr_schedule(step, steps, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = dataset.get_batch(batch_size, gen)
        _, losses = model(x, y)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses = {k: v.detach() for k, v in losses.items()}
        mv = moe_max_violation(model)
        for m in moes:
            if m.balance == "bias":
                m.update_bias()              # rule (6.5): no gradient involved
            else:
                m.load_accum.zero_()
        last = step == steps - 1
        if step % log_every == 0 or last:
            rec = {"step": step, "lr": cur_lr, "loss": float(losses["lm"]),
                   "ppl": math.exp(float(losses["lm"])), "mtp": float(losses["mtp"]),
                   "aux": float(losses["aux"]), "maxvio": mv,
                   "time": round(time.time() - t0, 2)}
            if val_dataset is not None and (step % eval_every == 0 or last):
                rec.update(evaluate_loss(model, val_dataset, batch_size))
            history.append(rec)
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
            if verbose and (step % (log_every * 10) == 0 or last):
                extra = f" val_ppl {rec['val_ppl']:.1f}" if "val_ppl" in rec else ""
                mvs = f" maxvio {mv:.2f}" if mv is not None else ""
                print(f"step {step:5d} lr {cur_lr:.2e} loss {rec['loss']:.3f} "
                      f"mtp {rec['mtp']:.3f}{mvs}{extra} [{rec['time']:.0f}s]")
            if callback:
                callback(step, rec)
    if log_f:
        log_f.close()
    model.eval()
    return model, history


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(model: TinySeekModel, out_dir: str | Path,
                    tokenizer_path: Optional[str | Path] = None) -> Path:
    """Write ``config.json``, ``model.pt`` (and ``tokenizer.json``) to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.to_json(out / "config.json")
    torch.save(model.state_dict(), out / "model.pt")
    if tokenizer_path is not None and Path(tokenizer_path) != out / "tokenizer.json":
        shutil.copy(tokenizer_path, out / "tokenizer.json")
    return out


def load_checkpoint(run_dir: str | Path) -> tuple[TinySeekModel, Optional[ByteBPETokenizer]]:
    """Load a model (in eval mode) and its tokenizer from a run directory."""
    d = Path(run_dir)
    cfg = TinySeekConfig.from_json(d / "config.json")
    model = TinySeekModel(cfg)
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    tok = ByteBPETokenizer.load(d / "tokenizer.json") if (d / "tokenizer.json").exists() else None
    return model, tok


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Pre-train TinySeek on the local corpus "
                                "with the DeepSeek-V3 LR schedule (Chapter 9).")
    p.add_argument("--preset", default="tiny_v3", choices=sorted(presets))
    p.add_argument("--config", help="JSON config file (overrides --preset)")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--balance", choices=["none", "aux", "bias"])
    p.add_argument("--mtp-lambda", type=float)
    p.add_argument("--fp8", action="store_true", help="emulate FP8 GEMMs (Chapter 10)")
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out", default="runs/v3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = (TinySeekConfig.from_json(args.config) if args.config else get_preset(args.preset))
    over = {}
    if args.balance:
        over["balance"] = args.balance
    if args.mtp_lambda is not None:
        over["mtp_lambda"] = args.mtp_lambda
    cfg = cfg.replace(**over)
    tok, data = prepare_data(args.data, cfg.vocab_size, args.seed)
    train_ds = PackedDataset(data["train"], args.seq_len)
    val_ds = PackedDataset(data["val"], args.seq_len)
    set_seed(args.seed)
    model = TinySeekModel(cfg)
    if args.fp8:
        from .fp8 import convert_to_fp8
        convert_to_fp8(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, hist = train(cfg, train_ds, args.steps, args.batch_size, args.lr, args.warmup,
                        seed=args.seed, model=model, val_dataset=val_ds,
                        log_path=out / "log.jsonl")
    if args.fp8:
        from .fp8 import convert_from_fp8
        convert_from_fp8(model)
    save_checkpoint(model, out, Path(args.data) / f"tokenizer_{cfg.vocab_size}.json")
    print(f"saved to {out}; final val ppl {hist[-1].get('val_ppl', float('nan')):.2f}")


if __name__ == "__main__":
    main()
