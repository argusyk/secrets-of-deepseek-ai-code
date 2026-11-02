"""Distilling reasoning from a teacher into a smaller student (Chapter 14).

The teacher is a TinySeek-V3 model after SFT/GRPO, the student a smaller
dense model with the same tokenizer. Three ways to train the student with
the same number of steps are compared:

* ``traces`` - the DeepSeek-R1 recipe (Section 14.4): the teacher samples
  several completions per prompt, only verified ones are kept (rejection
  sampling with the rule-based reward) and the student is fine-tuned on
  them (sequence-level distillation, formula (14.4));
* ``kd`` - token-level knowledge distillation on the same traces:
  L = alpha * tau^2 * KL(p_tau || q_tau) + (1 - alpha) * CE (formula (14.2));
* ``answer_only`` - a baseline without reasoning: the student sees only
  ``<answer>N</answer>`` targets, no chain of thought.

CLI::

    python -m tinyseek.distill --teacher runs/grpo --method traces --out runs/distill
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import TinySeekConfig, get_preset
from .data import arith_completion, make_arith_task
from .evaluate import complete_prompts
from .grpo import rule_reward
from .model import IGNORE_INDEX, TinySeekModel
from .sft import build_example, collate, sft
from .train import load_checkpoint, save_checkpoint, set_seed


def student_config(vocab_size: int = 2048) -> TinySeekConfig:
    """Default student: a dense MHA model with half the width and depth."""
    return get_preset("tiny_dense", vocab_size=vocab_size,
                      hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                      qk_nope_head_dim=32, v_head_dim=32, intermediate_size=192,
                      first_k_dense_replace=2).replace(name="tiny_student")


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            targets: torch.Tensor, tau: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """alpha * tau^2 * KL(p_tau || q_tau) + (1 - alpha) * CE(y, q)  (14.2),
    averaged over positions whose target is not -100."""
    mask = targets != IGNORE_INDEX
    s = student_logits[mask].float()
    t = teacher_logits[mask].float()
    kl = F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1),
                  log_target=True, reduction="batchmean")
    ce = F.cross_entropy(s, targets[mask])
    return alpha * tau * tau * kl + (1 - alpha) * ce


def collect_teacher_traces(teacher: TinySeekModel, tok, n_prompts: int = 2000,
                           samples: int = 4, temperature: float = 0.7,
                           digits: Sequence[int] = (1, 2, 3), seed: int = 0) -> dict:
    """Rejection sampling: keep teacher completions with reward 2 (correct and
    well formed), at most one per prompt. Returns the traces and statistics."""
    rng = np.random.default_rng(seed)
    tasks = [make_arith_task(rng, int(rng.choice(list(digits)))) for _ in range(n_prompts)]
    comps = complete_prompts(teacher, tok, [p for p, _ in tasks], 48, temperature,
                             samples, seed)
    traces, n_ok = [], 0
    for (p, a), cs in zip(tasks, comps):
        good = [c for c in cs if rule_reward(c, a) >= 2.0]
        n_ok += len(good)
        if good:
            traces.append((p, good[0]))
    return {"traces": traces, "kept_prompts": len(traces) / n_prompts,
            "sample_accuracy": n_ok / (n_prompts * samples)}


def kd_train(student: TinySeekModel, teacher: TinySeekModel, examples, pad_id: int,
             steps: int, batch_size: int = 32, lr: float = 2e-3, tau: float = 2.0,
             alpha: float = 0.5, seed: int = 0, verbose: bool = True,
             log_every: int = 10) -> list[dict]:
    """Token-level KD on tokenized examples (same tokenizer for both models)."""
    gen = set_seed(seed)
    student.train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    hist, t0 = [], time.time()
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = cur
        idx = torch.randint(0, len(examples), (batch_size,), generator=gen).tolist()
        x, y = collate([examples[i] for i in idx], pad_id)
        with torch.no_grad():
            t_logits, _ = teacher(x)
        s_logits, _ = student(x)
        loss = kd_loss(s_logits, t_logits, y, tau, alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss.detach()),
                         "time": round(time.time() - t0, 2)})
            if verbose and (step % (10 * log_every) == 0 or step == steps - 1):
                print(f"kd step {step:4d} loss {hist[-1]['loss']:.3f}")
    student.eval()
    return hist


def distill(teacher: TinySeekModel, tok, method: str = "traces", steps: int = 600,
            n_prompts: int = 2000, batch_size: int = 32, lr: float = 2e-3,
            tau: float = 2.0, alpha: float = 0.5, digits: Sequence[int] = (1, 2, 3),
            student_cfg: Optional[TinySeekConfig] = None, traces: Optional[list] = None,
            seed: int = 0, verbose: bool = True) -> tuple[TinySeekModel, dict]:
    """Train a fresh student with ``method`` in {traces, kd, answer_only}."""
    info: dict = {"method": method}
    if method in ("traces", "kd") and traces is None:
        col = collect_teacher_traces(teacher, tok, n_prompts, digits=digits, seed=seed)
        traces = col["traces"]
        info.update(kept_prompts=col["kept_prompts"], sample_accuracy=col["sample_accuracy"])
    if method == "answer_only":
        rng = np.random.default_rng(seed)
        tasks = [make_arith_task(rng, int(rng.choice(list(digits)))) for _ in range(n_prompts)]
        pairs = [(p, arith_completion(p, a, with_reasoning=False)) for p, a in tasks]
    else:
        pairs = traces
    examples = [build_example(tok, p, c) for p, c in pairs]
    info["examples"] = len(examples)
    set_seed(seed)
    student = TinySeekModel(student_cfg or student_config(teacher.config.vocab_size))
    if method == "kd":
        hist = kd_train(student, teacher, examples, tok.pad_id, steps, batch_size, lr, tau,
                        alpha, seed, verbose)
    else:
        hist = sft(student, examples, tok.pad_id, steps, batch_size, lr, seed=seed,
                   verbose=verbose)
    info["history"] = hist
    return student, info


def main(argv=None) -> None:
    from .evaluate import arith_accuracy
    from .model import count_parameters
    p = argparse.ArgumentParser(description="Distill arithmetic reasoning from a "
                                "TinySeek teacher into a small dense student (Chapter 14).")
    p.add_argument("--teacher", default="runs/grpo")
    p.add_argument("--method", default="traces", choices=["traces", "kd", "answer_only"])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--n-prompts", type=int, default=2000)
    p.add_argument("--tau", type=float, default=2.0, help="KD temperature")
    p.add_argument("--alpha", type=float, default=0.5, help="KD weight of the KL term")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--digits", default="1,2")
    p.add_argument("--out", default="runs/distill")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    digits = [int(d) for d in args.digits.split(",")]
    teacher, tok = load_checkpoint(args.teacher)
    student, info = distill(teacher, tok, args.method, args.steps, args.n_prompts,
                            lr=args.lr, tau=args.tau, alpha=args.alpha, digits=digits,
                            seed=args.seed)
    out = Path(args.out)
    save_checkpoint(student, out, Path(args.teacher) / "tokenizer.json")
    acc = arith_accuracy(student, tok, n=200, digits=digits)
    t_acc = arith_accuracy(teacher, tok, n=200, digits=digits)
    print(f"teacher: {count_parameters(teacher):,} params, accuracy {t_acc['accuracy']:.3f}")
    print(f"student ({args.method}): {count_parameters(student):,} params, "
          f"accuracy {acc['accuracy']:.3f}")
    (out / "info.json").write_text(json.dumps({k: v for k, v in info.items()
                                               if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
