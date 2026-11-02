"""Evaluation: perplexity, arithmetic accuracy, FIM pass@k (Chapters 9, 13, 17).

* `perplexity` - exp of the mean next-token cross-entropy (formulas (2.2)-(2.3)).
* `arith_accuracy` - share of correct answers and of well-formed outputs on
  fresh arithmetic problems, judged by the rule-based reward of Chapter 13.
* `pass_at_k` - the unbiased estimator 1 - C(n-c, k) / C(n, k) (17.1).
* `make_fim_tasks` / `fim_pass_at_k` - fill-in-the-middle tasks cut from
  held-out standard-library code: one identifier is hidden and the model
  must restore it exactly from the prefix and the suffix (PSM format).

CLI::

    python -m tinyseek.evaluate --run runs/v3 --ppl --arith --fim
"""
from __future__ import annotations

import argparse
import math
import re
from math import comb
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .data import DEFAULT_DATA_DIR, PackedDataset, load_corpus, make_arith_task
from .generate import generate_ids
from .grpo import _normalize, parse_completion, rule_reward
from .model import TinySeekModel


@torch.no_grad()
def perplexity(model: TinySeekModel, tokens, seq_len: int = 256, batch_size: int = 8,
               max_batches: int = 16) -> float:
    """exp(mean cross-entropy) over the first blocks of a token array."""
    ds = tokens if isinstance(tokens, PackedDataset) else PackedDataset(tokens, seq_len)
    model.eval()
    tot, n = 0.0, 0
    for x, y in ds.iter_batches(batch_size, max_batches):
        _, losses = model(x, y)
        tot += float(losses["lm"])
        n += 1
    return math.exp(tot / n)


@torch.no_grad()
def complete_prompts(model: TinySeekModel, tok, prompts: Sequence[str],
                     max_new_tokens: int = 48, temperature: float = 0.0,
                     samples: int = 1, seed: int = 0, top_k: Optional[int] = None
                     ) -> list[list[str]]:
    """``samples`` completions for each prompt (prompts of equal token length
    are batched together)."""
    gen = torch.Generator().manual_seed(seed)
    enc = [tok.encode(p) for p in prompts]
    out: list[list[str]] = [[] for _ in prompts]
    by_len: dict[int, list[int]] = {}
    for i, e in enumerate(enc):
        by_len.setdefault(len(e), []).append(i)
    for _, idxs in sorted(by_len.items()):
        for c in range(0, len(idxs), max(1, 64 // samples)):
            chunk = idxs[c:c + max(1, 64 // samples)]
            ids = torch.tensor([enc[i] for i in chunk for _ in range(samples)])
            new = generate_ids(model, ids, max_new_tokens, temperature, top_k, None, gen,
                               True, tok.eos_id)
            for r, row in enumerate(new.tolist()):
                if tok.eos_id in row:
                    row = row[:row.index(tok.eos_id)]
                out[chunk[r // samples]].append(tok.decode(row))
    return out


ANSWER_RE = re.compile(r"<answer>(.+?)</answer>", re.DOTALL)


def arith_accuracy(model: TinySeekModel, tok, n: int = 200, digits: Sequence[int] = (1, 2, 3),
                   seed: int = 12345, temperature: float = 0.0, samples: int = 1,
                   max_new_tokens: int = 48, ops: str = "+-") -> dict:
    """Accuracy (the ``<answer>`` equals the reference, with or without a
    ``<think>`` part), format rate (the full R1 template), mean rule-based
    reward and mean completion length."""
    rng = np.random.default_rng(seed)
    tasks = [make_arith_task(rng, int(rng.choice(list(digits))), ops) for _ in range(n)]
    comps = complete_prompts(model, tok, [p for p, _ in tasks], max_new_tokens,
                             temperature, samples, seed)
    acc, fmt, rew, length = [], [], [], []
    for (p, a), cs in zip(tasks, comps):
        for c in cs:
            r = rule_reward(c, a)
            fmt.append(parse_completion(c) is not None)
            m = ANSWER_RE.search(c)                 # correctness, with or without <think>
            acc.append(bool(m) and _normalize(m.group(1)) == _normalize(a))
            rew.append(r)
            length.append(len(tok.encode(c)))
    return {"accuracy": float(np.mean(acc)), "format": float(np.mean(fmt)),
            "reward": float(np.mean(rew)), "length": float(np.mean(length)), "n": len(acc)}


# ---------------------------------------------------------------------- FIM
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (formula (17.1))."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def make_fim_tasks(data_dir: str | Path = DEFAULT_DATA_DIR, n: int = 40, seed: int = 0,
                   prefix_chars: int = 400, suffix_chars: int = 150) -> list[dict]:
    """FIM tasks from held-out code: hide one identifier (>= 3 characters)."""
    docs = [r["text"] for r in load_corpus(data_dir)
            if r["source"] == "code" and r["split"] == "val"]
    rng = np.random.default_rng(seed)
    tasks = []
    word = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z_0-9]{2,15}(?!\w)")
    attempts = 0
    while len(tasks) < n and docs and attempts < 50 * n:
        attempts += 1
        d = docs[int(rng.integers(len(docs)))]
        matches = [m for m in word.finditer(d) if m.start() > 200]
        if not matches:
            continue
        m = matches[int(rng.integers(len(matches)))]
        tasks.append({"prefix": d[max(0, m.start() - prefix_chars):m.start()],
                      "middle": m.group(0),
                      "suffix": d[m.end():m.end() + suffix_chars]})
    return tasks


def fim_prompt(task: dict) -> str:
    return f"<|fim_begin|>{task['prefix']}<|fim_hole|>{task['suffix']}<|fim_end|>"


def fim_pass_at_k(model: TinySeekModel, tok, tasks: Sequence[dict], n: int = 10,
                  ks: Sequence[int] = (1, 5, 10), temperature: float = 0.8,
                  seed: int = 0) -> dict:
    """Mean pass@k over FIM tasks (exact match of the hidden identifier)."""
    comps = complete_prompts(model, tok, [fim_prompt(t) for t in tasks], 8, temperature,
                             n, seed)
    res = {f"pass@{k}": [] for k in ks}
    for t, cs in zip(tasks, comps):
        c = sum(x.strip() == t["middle"] for x in cs)
        for k in ks:
            res[f"pass@{k}"].append(pass_at_k(n, c, k))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main(argv=None) -> None:
    from .data import prepare_data
    from .train import load_checkpoint
    p = argparse.ArgumentParser(description="Evaluate a TinySeek checkpoint "
                                "(Chapters 9, 13, 17).")
    p.add_argument("--run", default="runs/v3")
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--ppl", action="store_true", help="validation perplexity by source")
    p.add_argument("--arith", action="store_true", help="arithmetic accuracy")
    p.add_argument("--fim", action="store_true", help="FIM pass@k on held-out code")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    model, tok = load_checkpoint(args.run)
    if not (args.ppl or args.arith or args.fim):
        args.ppl = args.arith = True
    if args.ppl:
        _, data = prepare_data(args.data, model.config.vocab_size, args.seed, verbose=False)
        print(f"val ppl (all): {perplexity(model, data['val']):.2f}")
        for src, arr in data["val_by_source"].items():
            if len(arr) > 300:
                print(f"val ppl ({src}): {perplexity(model, arr, max_batches=4):.2f}")
    if args.arith:
        r = arith_accuracy(model, tok, n=args.n)
        print(f"arithmetic: accuracy {r['accuracy']:.3f}, format {r['format']:.3f}, "
              f"mean length {r['length']:.1f}")
    if args.fim:
        tasks = make_fim_tasks(args.data, n=40, seed=args.seed)
        print("FIM:", {k: round(v, 3) for k, v in fim_pass_at_k(model, tok, tasks).items()})


if __name__ == "__main__":
    main()
