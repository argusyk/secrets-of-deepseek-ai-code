"""An evaluation harness: a labelled set, a model under test, metrics.

By default the "model" is a deterministic mock that classifies snippets with
simple rules and is deliberately wrong or malformed on some items (on this
set it scores 0.50), so every branch of the harness is exercised offline.
With --base-url / --model the same harness evaluates a real backend through
the OpenAI-compatible client: the DeepSeek API or a local TinySeek server.

Metrics: JSON validity, accuracy with its standard error (formula (18.2)),
the Wald and the Wilson 95% intervals (formula (18.3)), a list of mistakes,
the mean latency and the token usage of a real backend.

See Chapter 18 of "Secrets of DeepSeek AI".

Run:
    python eval_harness.py                                    # offline mock
    DEEPSEEK_API_KEY=sk-... python eval_harness.py --base-url api
    python eval_harness.py --base-url http://127.0.0.1:8000/v1 --model tinyseek
"""
import argparse
import ast
import json
import math
import random
import sys
import time
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
                isinstance(d, (ast.List, ast.Dict, ast.Set))
                for d in node.args.defaults):
            return "mutable-default"
        if isinstance(node, ast.Call) and \
                getattr(node.func, "id", None) in ("eval", "exec"):
            return "eval-exec"
        if isinstance(node, ast.Compare) and any(
                isinstance(c, ast.Constant) and c.value is None
                for c in node.comparators):
            return "eq-none"
    return "none"


class MockModel:
    """Rule-based labels spoiled on purpose: some answers are relabelled
    'other', some are truncated JSON (accuracy 0.50 on DATASET)."""

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


class ApiModel:
    """A real backend behind the OpenAI-compatible client (json_mode.py)."""

    def __init__(self, client, model):
        self.client, self.model = client, model
        self.seconds, self.calls, self.tokens = 0.0, 0, 0

    def __call__(self, code):
        from json_mode import SYSTEM
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=200,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": code}],
            response_format={"type": "json_object"})
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.tokens += getattr(resp.usage, "total_tokens", 0) or 0
        return resp.choices[0].message.content or ""


def wilson_interval(p, n, z=1.96):
    """Wilson score interval for a proportion p measured on n items."""
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half / (1 + z * z / n), center + half / (1 + z * z / n)


def evaluate(model, data):
    n = len(data)
    valid = correct = 0
    mistakes = []
    for code, gold in data:
        raw = model(code)
        try:
            pred = json.loads(raw)["category"]
            valid += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pred = "<invalid>"
        if pred == gold:
            correct += 1
        else:
            mistakes.append((gold, pred))
    p = correct / n
    se = math.sqrt(p * (1 - p) / n)
    return {"n": n, "json_valid": valid / n, "accuracy": p, "se": se,
            "wald95": (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)),
            "wilson95": wilson_interval(p, n), "mistakes": mistakes}


def pass_at_k(n, c, k):
    """Unbiased pass@k, formula (17.1) of Chapter 17."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", help="evaluate a real backend; "
                    "'api' means the DeepSeek API")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    if a.base_url:
        from client import DEEPSEEK_URL, api_errors, make_client
        url = DEEPSEEK_URL if a.base_url == "api" else a.base_url
        model = ApiModel(make_client(url, timeout=a.timeout), a.model)
        try:
            r = evaluate(model, DATASET)
        except api_errors() as e:
            sys.exit(f"request failed: {type(e).__name__}: {e}")
    else:
        model = MockModel()
        r = evaluate(model, DATASET)
    lo, hi = r["wald95"]
    wl, wh = r["wilson95"]
    print(f"items: {r['n']}  json valid: {r['json_valid']:.0%}")
    print(f"accuracy: {r['accuracy']:.2f} +/- {r['se']:.2f} "
          f"(95% Wald {lo:.2f}..{hi:.2f}, Wilson {wl:.2f}..{wh:.2f})")
    for gold, pred in r["mistakes"]:
        print(f"  expected {gold:16s} got {pred[:40]!r}")
    if isinstance(model, ApiModel):
        print(f"mean latency {model.seconds / model.calls:.2f}s, "
              f"tokens {model.tokens}")
    else:
        print("pass@1, pass@5 for n=10 samples with c=3 correct:",
              round(pass_at_k(10, 3, 1), 3), round(pass_at_k(10, 3, 5), 3))
        for n_items in (12, 100, 400):
            se = math.sqrt(0.8 * 0.2 / n_items)
            print(f"SE at p=0.8 with {n_items:4d} items: {se:.3f}")


if __name__ == "__main__":
    main()
