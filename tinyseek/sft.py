"""Supervised fine-tuning on arithmetic instructions with prompt masking (Chapter 12).

Every example is ``prompt + completion + <|eos|>`` where the completion is
``<think>column-wise steps</think><answer>N</answer>``. The loss (12.1) is
averaged over completion tokens only: positions that predict prompt tokens
get the label -100 and are ignored, so the model learns to answer rather
than to repeat questions.

CLI::

    python -m tinyseek.sft --init runs/v3 --steps 400 --out runs/sft
    python -m tinyseek.sft --preset tiny_v3 --steps 600 --out runs/sft_scratch
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .config import get_preset, presets
from .data import DEFAULT_DATA_DIR, arith_completion, make_arith_task, prepare_data
from .model import IGNORE_INDEX, TinySeekModel
from .train import load_checkpoint, save_checkpoint, set_seed


def build_example(tok, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    """Token ids and labels for one example; prompt positions are masked."""
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion) + [tok.eos_id]
    ids = p_ids + c_ids
    labels = [IGNORE_INDEX] * (len(p_ids) - 1) + c_ids     # label[t] = ids[t+1]
    return ids[:-1], labels


def collate(examples: Sequence[tuple[list[int], list[int]]], pad_id: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of (ids, labels) to a batch; padding is ignored."""
    T = max(len(i) for i, _ in examples)
    x = torch.full((len(examples), T), pad_id, dtype=torch.long)
    y = torch.full((len(examples), T), IGNORE_INDEX, dtype=torch.long)
    for r, (ids, lab) in enumerate(examples):
        x[r, :len(ids)] = torch.tensor(ids)
        y[r, :len(lab)] = torch.tensor(lab)
    return x, y


def arith_sft_examples(tok, n: int, digits: Sequence[int] = (1, 2, 3), seed: int = 0,
                       with_reasoning: bool = True, ops: str = "+-"
                       ) -> list[tuple[list[int], list[int]]]:
    """``n`` tokenized arithmetic instructions with reference completions."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        prompt, ans = make_arith_task(rng, int(rng.choice(list(digits))), ops)
        out.append(build_example(tok, prompt, arith_completion(prompt, ans, with_reasoning)))
    return out


def sft(model: TinySeekModel, examples: Sequence[tuple[list[int], list[int]]],
        pad_id: int, steps: int, batch_size: int = 32, lr: float = 1e-3,
        warmup: int = 20, seed: int = 0, log_path: Optional[str | Path] = None,
        callback: Optional[Callable[[int, dict], None]] = None,
        verbose: bool = True, log_every: int = 10) -> list[dict]:
    """Fine-tune ``model`` on ``examples`` (cosine LR after a short warm-up).

    The MoE balancing biases keep being updated, as during pre-training.
    Returns the logged records (``step, loss, mtp, time``).
    """
    gen = set_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    hist, t0 = [], time.time()
    log_f = open(log_path, "w", encoding="utf-8") if log_path else None
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = cur
        idx = torch.randint(0, len(examples), (batch_size,), generator=gen).tolist()
        x, y = collate([examples[i] for i in idx], pad_id)
        _, losses = model(x, y)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses = {k: v.detach() for k, v in losses.items()}
        for m in model.moe_layers():
            if m.balance == "bias":
                m.update_bias()
            else:
                m.load_accum.zero_()
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "lr": cur, "loss": float(losses["lm"]),
                   "mtp": float(losses["mtp"]), "time": round(time.time() - t0, 2)}
            hist.append(rec)
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
            if callback:
                callback(step, rec)
            if verbose and (step % (log_every * 10) == 0 or step == steps - 1):
                print(f"sft step {step:4d} loss {rec['loss']:.3f} [{rec['time']:.0f}s]")
    if log_f:
        log_f.close()
    model.eval()
    return hist


def main(argv=None) -> None:
    from .evaluate import arith_accuracy
    p = argparse.ArgumentParser(description="SFT with prompt masking on arithmetic "
                                "instructions (Chapter 12).")
    p.add_argument("--init", help="checkpoint to start from (e.g. runs/v3)")
    p.add_argument("--preset", default="tiny_v3", choices=sorted(presets),
                   help="architecture when training from scratch")
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-examples", type=int, default=20000)
    p.add_argument("--digits", default="1,2", help="operand lengths, e.g. 1,2,3")
    p.add_argument("--no-reasoning", action="store_true", help="answers without <think>")
    p.add_argument("--out", default="runs/sft")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    digits = [int(d) for d in args.digits.split(",")]
    if args.init:
        model, tok = load_checkpoint(args.init)
        tok_path = Path(args.init) / "tokenizer.json"
    else:
        cfg = get_preset(args.preset)
        tok, _ = prepare_data(args.data, cfg.vocab_size, args.seed)
        tok_path = Path(args.data) / f"tokenizer_{cfg.vocab_size}.json"
        set_seed(args.seed)
        model = TinySeekModel(cfg)
    ex = arith_sft_examples(tok, args.n_examples, digits, args.seed + 1,
                            not args.no_reasoning)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sft(model, ex, tok.pad_id, args.steps, args.batch_size, args.lr, seed=args.seed,
        log_path=out / "log.jsonl")
    save_checkpoint(model, out, tok_path)
    acc = arith_accuracy(model, tok, n=200, digits=digits, seed=12345)
    print(f"greedy accuracy {acc['accuracy']:.3f}, format {acc['format']:.3f}")


if __name__ == "__main__":
    main()
