"""An offline evaluation harness: a labelled set, a model under test, metrics.

The "model" here is a deterministic mock that classifies snippets with simple
rules and is deliberately wrong or malformed on a few items, so every branch
of the harness is exercised. Swap MockModel for a real client (see
json_mode.py) to evaluate DeepSeek itself.

Metrics: JSON validity, accuracy with its standard error and a 95% interval,
a confusion list, and the unbiased pass@k estimator.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (offline):
    python eval_harness.py
"""
import ast
import json
import math
import random
from math import comb

DATASET = [
    ("def f(x, acc=[]):\n    acc.append(x)\n    return acc", "mutable-default"),
    ("try:\n    run()\nexcept:\n    pass", "bare-except"),
    ("cfg = eval(open('c.txt').read())", "eval-exec"),
    ("if user == None:\n    login()", "eq-none"),
    ("def area(r):\n    return 3.14159 * r * r", "none"),
    ("def g(d={}):\n    return d", "mutable-default"),
    ("exec(payload)", "eval-exec"),
    ("total = sum(x for x in xs)", "none"),
    ("try:\n    x = int(s)\nexcept ValueError:\n    x = 0", "none"),
    ("if x != None and y:\n    go()", "eq-none"),
    ("try:\n    close()\nexcept:\n    log()", "bare-except"),
    ("names = [u.name for u in users]", "none"),
]


def rule_label(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            return "bare-except"
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in node.args.defaults):
            return "mutable-default"
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("eval", "exec"):
            return "eval-exec"
        if isinstance(node, ast.Compare) and any(
                isinstance(c, ast.Constant) and c.value is None for c in node.comparators):
            return "eq-none"
    return "none"


class MockModel:
    """Right most of the time; returns broken JSON or a wrong label sometimes."""

    def __init__(self, seed=7, p_wrong=0.15, p_malformed=0.1):
        self.rng = random.Random(seed)
        self.p_wrong, self.p_malformed = p_wrong, p_malformed

    def __call__(self, code):
        if self.rng.random() < self.p_malformed:
            return '{"category": "none", '          # truncated output
        label = rule_label(code)
        if self.rng.random() < self.p_wrong:
            label = "other"
        return json.dumps({"has_bug": label != "none", "category": label})


def evaluate(model, data):
    n = len(data); valid = correct = 0; mistakes = []
    for code, gold in data:
        raw = model(code)
        try:
            pred = json.loads(raw)["category"]; valid += 1
        except (json.JSONDecodeError, KeyError):
            pred = "<invalid>"
        if pred == gold:
            correct += 1
        else:
            mistakes.append((gold, pred))
    p = correct / n
    se = math.sqrt(p * (1 - p) / n)
    return {"n": n, "json_valid": valid / n, "accuracy": p, "se": se,
            "ci95": (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)),
            "mistakes": mistakes}


def pass_at_k(n, c, k):
    """Unbiased pass@k: probability that at least one of k samples (drawn from
    n generated, c of them correct) is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


if __name__ == "__main__":
    r = evaluate(MockModel(), DATASET)
    print(f"items: {r['n']}  json valid: {r['json_valid']:.0%}")
    print(f"accuracy: {r['accuracy']:.2f} +/- {r['se']:.2f} (95% CI {r['ci95'][0]:.2f}..{r['ci95'][1]:.2f})")
    for gold, pred in r["mistakes"]:
        print(f"  expected {gold:16s} got {pred}")
    print("pass@1, pass@5 for n=10 samples with c=3 correct:",
          round(pass_at_k(10, 3, 1), 3), round(pass_at_k(10, 3, 5), 3))
    for n_items in (12, 100, 400):
        print(f"SE at p=0.8 with {n_items:4d} items: {math.sqrt(0.8 * 0.2 / n_items):.3f}")
