"""Unbiased pass@k estimator for code benchmarks such as HumanEval.

For each problem we draw n >= k samples, c of which pass the unit tests.
pass@k = 1 - C(n - c, k) / C(n, k), averaged over problems. The product form
below avoids huge binomial coefficients. See Chapter 17 of
"Secrets of DeepSeek AI".

Run:
    python pass_at_k.py
"""
import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    # 1 - prod_{i=n-c+1}^{n} (1 - k / i)
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def naive_estimate(n: int, c: int, k: int, rng, trials: int = 20000) -> float:
    """Monte Carlo check: draw k of the n samples without replacement."""
    outcomes = np.array([1] * c + [0] * (n - c))
    hits = sum(outcomes[rng.choice(n, size=k, replace=False)].any() for _ in range(trials))
    return hits / trials


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20
    results = [(n, c) for c in (0, 1, 3, 10, 20)]
    print(" c   pass@1  pass@5  pass@10  (Monte Carlo pass@5)")
    for n, c in results:
        print(f"{c:2d}   {pass_at_k(n, c, 1):.3f}   {pass_at_k(n, c, 5):.3f}   "
              f"{pass_at_k(n, c, 10):.3f}    {naive_estimate(n, c, 5, rng):.3f}")
    bench = [pass_at_k(n, c, 1) for _, c in results]
    print(f"\nbenchmark pass@1 over these five problems: {np.mean(bench):.3f}")
