"""GRPO with rule-based rewards on arithmetic (Chapter 13).

* `rule_reward` - format reward (the completion must be
  ``<think>...</think><answer>...</answer>``) plus accuracy reward (the
  answer must equal the reference), formula (13.6); range 0..2.
* `grpo_advantages` - A_i = (r_i - mean(r)) / std(r) inside each group (13.3).
* `kl_k3` - the unbiased non-negative KL estimator (13.4).
* `grpo_loss` - the clipped surrogate with per-token ratios, averaged over
  the tokens of each completion and over the group (13.5), minus beta * KL.
* `grpo_train` - the loop: for every prompt sample a group of G completions
  from the current policy, score them, compute group-relative advantages and
  take ``inner_steps`` gradient steps. No critic (value model) is involved.

CLI::

    python -m tinyseek.grpo --init runs/sft --steps 40 --out runs/grpo
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_arith_task

TEMPLATE = re.compile(r"^\s*<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$", re.DOTALL)


# -------------------------------------------------------------------- rewards
def parse_completion(completion: str) -> Optional[tuple[str, str]]:
    """(reasoning, answer) if the completion follows the template, else None."""
    m = TEMPLATE.match(completion)
    return (m.group(1), m.group(2)) if m else None


def _normalize(ans: str) -> str:
    ans = ans.strip().strip(".").replace(" ", "").replace(",", ".")
    try:
        return repr(float(ans))
    except ValueError:
        return ans.lower()


def rule_reward(completion: str, reference: str) -> float:
    """r = r_format + r_acc (formula (13.6)); an unparseable answer earns 0."""
    parsed = parse_completion(completion)
    if parsed is None:
        return 0.0
    return 1.0 + (1.0 if _normalize(parsed[1]) == _normalize(reference) else 0.0)


# ---------------------------------------------------------------- GRPO maths
def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantages (13.3). ``rewards`` has shape (groups, G);
    inside every group the result has zero mean and unit (population) std,
    or is all zeros when every member got the same reward."""
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.dim() == 1:
        r = r[None]
    mean = r.mean(-1, keepdim=True)
    std = r.std(-1, unbiased=False, keepdim=True)
    return (r - mean) / (std + eps)


def kl_k3(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """pi_ref/pi - log(pi_ref/pi) - 1 per token (13.4); always >= 0."""
    d = logp_ref - logp
    return d.exp() - d - 1.0


def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor,
              advantages: torch.Tensor, mask: torch.Tensor, eps_clip: float = 0.2,
              beta: float = 0.04) -> tuple[torch.Tensor, dict]:
    """Negative GRPO objective (13.5) for a batch of completions.

    All tensors are (B, T) per-token values except ``advantages`` (B,).
    ``mask`` marks completion tokens. Returns (loss, stats).
    """
    ratio = (logp - logp_old).exp()
    a = advantages[:, None]
    surr = torch.minimum(ratio * a, ratio.clamp(1 - eps_clip, 1 + eps_clip) * a)
    kl = kl_k3(logp, logp_ref)
    per_tok = (surr - beta * kl) * mask
    per_seq = per_tok.sum(-1) / mask.sum(-1).clamp_min(1)
    loss = -per_seq.mean()
    clipped = ((ratio - 1).abs() > eps_clip).float() * mask
    kl, clipped = kl.detach(), clipped.detach()
    stats = {"kl": float((kl * mask).sum() / mask.sum().clamp_min(1)),
             "clip_frac": float(clipped.sum() / mask.sum().clamp_min(1))}
    return loss, stats


def token_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    """log pi(ids[:, t+1] | ids[:, :t+1]) for t = 0..T-2, shape (B, T-1)."""
    logits, _ = model(ids[:, :-1])
    return torch.gather(F.log_softmax(logits.float(), -1), -1, ids[:, 1:, None]).squeeze(-1)


# ---------------------------------------------------------------- training loop
@torch.no_grad()
def sample_groups(model, tok, prompts: Sequence[str], group_size: int, max_new_tokens: int,
                  temperature: float, generator: torch.Generator) -> list:
    """G completions for each prompt. Prompts with the same token length are
    generated in one batch (the model has no padding mask, so lengths must
    match). Returns, per prompt, (batch ids, completion mask, texts)."""
    from .generate import generate_ids
    enc = [tok.encode(p) for p in prompts]
    rows: list = [None] * len(prompts)
    buckets: dict[int, list[int]] = {}
    for i, e in enumerate(enc):
        buckets.setdefault(len(e), []).append(i)
    for _, idxs in sorted(buckets.items()):
        ids = torch.tensor([enc[i] for i in idxs for _ in range(group_size)])
        new = generate_ids(model, ids, max_new_tokens, temperature, None, None, generator,
                           True, tok.eos_id).tolist()
        for k, i in enumerate(idxs):
            rows[i] = new[k * group_size:(k + 1) * group_size]
    out = []
    for p_ids, group in zip(enc, rows):
        comp, texts = [], []
        for row in group:
            if tok.eos_id in row:
                row = row[:row.index(tok.eos_id) + 1]   # keep eos: learn to stop
            comp.append(row)
            texts.append(tok.decode([t for t in row if t != tok.eos_id]))
        T = len(p_ids) + max(len(r) for r in comp)
        batch = torch.full((group_size, T), tok.pad_id, dtype=torch.long)
        mask = torch.zeros(group_size, T - 1)
        for j, r in enumerate(comp):
            batch[j, :len(p_ids) + len(r)] = torch.tensor(p_ids + r)
            mask[j, len(p_ids) - 1:len(p_ids) - 1 + len(r)] = 1.0   # predicts completion
        out.append((batch, mask, texts))
    return out


def sample_group(model, tok, prompt: str, group_size: int, max_new_tokens: int,
                 temperature: float, generator: torch.Generator):
    """G completions of one prompt: returns (batch ids, completion mask, texts)."""
    return sample_groups(model, tok, [prompt], group_size, max_new_tokens, temperature,
                         generator)[0]


def grpo_train(model, tok, steps: int = 40, prompts_per_step: int = 16, group_size: int = 8,
               lr: float = 2e-5, beta: float = 0.04, eps_clip: float = 0.2,
               inner_steps: int = 2, temperature: float = 1.0, max_new_tokens: int = 40,
               digits: Sequence[int] = (1, 2, 3), seed: int = 0,
               log_path: Optional[str | Path] = None,
               callback: Optional[Callable[[int, dict], None]] = None,
               verbose: bool = True) -> list[dict]:
    """Run GRPO on arithmetic; returns per-step records with the mean reward,
    accuracy, format rate, completion length, KL and clipping share."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed + 777)
    ref = copy.deepcopy(model).eval()                    # frozen pi_ref
    for p in ref.parameters():
        p.requires_grad_(False)
    model.eval()                                         # no MoE bias updates during RL
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    log_f = open(log_path, "w", encoding="utf-8") if log_path else None
    hist, t0 = [], time.time()
    for step in range(steps):
        groups = []
        rewards_all, lens = [], []
        tasks = [make_arith_task(rng, int(rng.choice(list(digits))))
                 for _ in range(prompts_per_step)]
        sampled = sample_groups(model, tok, [p for p, _ in tasks], group_size,
                                max_new_tokens, temperature, gen)
        for (prompt, answer), (batch, mask, texts) in zip(tasks, sampled):
            r = torch.tensor([rule_reward(t, answer) for t in texts])
            adv = grpo_advantages(r)[0]
            with torch.no_grad():
                old = token_logprobs(model, batch)
                refp = token_logprobs(ref, batch)
            groups.append((batch, mask, adv, old, refp))
            rewards_all.append(r)
            lens.append(mask.sum(-1))
        stats_acc = {"kl": 0.0, "clip_frac": 0.0}
        for _ in range(inner_steps):
            opt.zero_grad(set_to_none=True)
            for batch, mask, adv, old, refp in groups:
                if float(adv.abs().sum()) == 0.0 and beta == 0.0:
                    continue                             # no signal in this group
                logp = token_logprobs(model, batch)
                loss, st = grpo_loss(logp, old, refp, adv, mask, eps_clip, beta)
                (loss / len(groups)).backward()
                for k in stats_acc:
                    stats_acc[k] += st[k] / (len(groups) * inner_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        r = torch.cat(rewards_all)
        rec = {"step": step, "reward": float(r.mean()), "accuracy": float((r >= 2).float().mean()),
               "format": float((r >= 1).float().mean()),
               "length": float(torch.cat(lens).mean()), **stats_acc,
               "time": round(time.time() - t0, 2)}
        hist.append(rec)
        if log_f:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        if callback:
            callback(step, rec)
        if verbose and (step % 10 == 0 or step == steps - 1):
            print(f"grpo step {step:3d} reward {rec['reward']:.3f} acc {rec['accuracy']:.3f} "
                  f"len {rec['length']:.1f} kl {rec['kl']:.4f} [{rec['time']:.0f}s]")
    if log_f:
        log_f.close()
    model.eval()
    return hist


def main(argv=None) -> None:
    from .evaluate import arith_accuracy
    from .train import load_checkpoint, save_checkpoint
    p = argparse.ArgumentParser(description="GRPO with rule-based rewards on "
                                "arithmetic (Chapter 13).")
    p.add_argument("--init", default="runs/sft", help="SFT checkpoint to start from")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8, help="G")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="small: Adam moves every weight by ~lr per step, and RL gradients are noisy")
    p.add_argument("--beta", type=float, default=0.04, help="KL weight")
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--inner-steps", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--digits", default="1,2")
    p.add_argument("--out", default="runs/grpo")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    digits = [int(d) for d in args.digits.split(",")]
    model, tok = load_checkpoint(args.init)
    before = arith_accuracy(model, tok, n=200, digits=digits, temperature=1.0, seed=999)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grpo_train(model, tok, args.steps, args.prompts_per_step, args.group_size, args.lr,
               args.beta, args.eps_clip, args.inner_steps, args.temperature, digits=digits,
               seed=args.seed, log_path=out / "log.jsonl")
    after = arith_accuracy(model, tok, n=200, digits=digits, temperature=1.0, seed=999)
    save_checkpoint(model, out, Path(args.init) / "tokenizer.json")
    print(f"sampled (tau=1) reward {before['reward']:.3f} -> {after['reward']:.3f}, "
          f"accuracy {before['accuracy']:.3f} -> {after['accuracy']:.3f}")


if __name__ == "__main__":
    main()
