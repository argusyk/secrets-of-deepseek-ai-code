"""Build an SFT dataset for distillation from teacher reasoning traces.

Teaching code for Chapter 14 of "Secrets of DeepSeek AI". It mimics, on
synthetic arithmetic problems, the recipe used to distil DeepSeek-R1 into
dense models: sample several traces per prompt from the teacher, keep only
the ones that pass rule-based filters (format, correct answer, a single
language, bounded length, no duplicates), then mix in non-reasoning data.

The "teacher" here is a mock that sometimes errs on purpose, so the filters
have something to do. Runs offline; no model or API key needed.

Run:
    python make_sft_dataset.py                 # print statistics and a sample
    python make_sft_dataset.py --out sft.jsonl # also write the dataset
"""
import argparse
import json
import random
import re

THINK = re.compile(r"^<think>\n(.+?)\n</think>\nAnswer: (-?\d+)$", re.S)


def make_prompt(rng):
    a, b, c, d = (rng.randint(2, 30) for _ in range(4))
    expr = f"({a} + {b}) * {c} - {d}"
    return {"prompt": f"Compute {expr}. Think step by step.", "expr": expr,
            "answer": (a + b) * c - d}


def mock_teacher(item, rng):
    """One sampled trace. Injects the failure modes real samples show."""
    a, b, c, d = map(int, re.findall(r"\d+", item["expr"]))
    s = a + b
    if rng.random() < 0.20:                       # arithmetic slip
        s += rng.choice([-1, 1])
    p = s * c
    ans = p - d
    steps = [f"First, {a} + {b} = {s}.", f"Then {s} * {c} = {p}.", f"Finally, {p} - {d} = {ans}."]
    if rng.random() < 0.10:                       # overthinking
        steps += ["Wait, let me re-check the first step."] * rng.randint(20, 40)
    if rng.random() < 0.08:                       # language mixing
        steps.insert(1, "所以我们继续计算。")
    body = "\n".join(steps)
    if rng.random() < 0.05:                       # broken format
        return f"<think>\n{body}\nAnswer: {ans}"
    return f"<think>\n{body}\n</think>\nAnswer: {ans}"


def passes_filters(item, trace, max_words=120):
    m = THINK.match(trace)
    if not m:
        return False, "format"
    if int(m.group(2)) != item["answer"]:
        return False, "wrong answer"
    if not trace.isascii():
        return False, "language mixing"
    if len(trace.split()) > max_words:
        return False, "too long"
    return True, "ok"


def build(n_prompts=2000, k=4, keep_per_prompt=2, general_ratio=1 / 3, seed=0):
    rng = random.Random(seed)
    data, seen, stats = [], set(), {}
    for _ in range(n_prompts):
        item = make_prompt(rng)
        kept = 0
        for _ in range(k):                        # rejection sampling
            trace = mock_teacher(item, rng)
            ok, why = passes_filters(item, trace)
            stats[why] = stats.get(why, 0) + 1
            if ok and trace not in seen and kept < keep_per_prompt:
                seen.add(trace)
                data.append({"prompt": item["prompt"], "response": trace, "kind": "reasoning"})
                kept += 1
    n_general = int(len(data) * general_ratio)    # about 600K : 200K in R1
    for i in range(n_general):
        data.append({"prompt": f"Write a one-line greeting number {i}.",
                     "response": f"Hello, and have a good day ({i}).", "kind": "general"})
    rng.shuffle(data)
    return data, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the dataset as JSONL")
    args = ap.parse_args()
    data, stats = build()
    total = sum(stats.values())
    print(f"sampled traces: {total}")
    for why, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {why:16s} {n:6d}  ({100 * n / total:4.1f}%)")
    kinds = {}
    for r in data:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"dataset: {len(data)} examples {kinds}")
    print("\nexample:\n" + next(r for r in data if r["kind"] == "reasoning")["response"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwritten: {args.out}")
