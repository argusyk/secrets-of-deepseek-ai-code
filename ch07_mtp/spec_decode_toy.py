"""Speculative decoding with an MTP draft: a toy simulation in NumPy.

Two experiments (see Chapter 7, Section 7.5 and Listing 7.2 of "Secrets of
DeepSeek AI"):

1. Throughput. A draft of D tokens is verified in one pass of the main
   model; the j-th draft token survives only if all earlier ones were
   accepted. With a per-token acceptance rate alpha the expected number of
   tokens per pass is 1 + alpha + ... + alpha^D (formula (7.10)). For the
   depth-1 MTP draft of DeepSeek-V3 (alpha = 0.85-0.90) this is 1.85-1.9
   tokens per pass before overheads.

2. Exactness. The acceptance rule min(1, p/q) plus resampling from the
   normalized residual max(0, p - q) keeps the output distribution equal to
   the main model's distribution p, however poor the draft distribution q
   is. The acceptance probability equals sum(min(p, q)).

Run:
    python spec_decode_toy.py
"""
import numpy as np

rng = np.random.default_rng(42)


def expected_tokens(alpha, depth=1):
    """Analytic mean tokens per pass: sum of alpha^j, j = 0..depth."""
    return sum(alpha ** j for j in range(depth + 1))


def tokens_per_step(alpha, depth=1, steps=200_000):
    """Monte Carlo: accepted draft prefix + one main-model token."""
    acc = rng.random((steps, depth)) < alpha
    # number of draft tokens accepted before the first rejection
    prefix = np.cumprod(acc, axis=1).sum(axis=1)
    return 1.0 + prefix.mean()


def speculative_sample(p, q):
    """Draw one token distributed exactly as p, using a draft from q."""
    x = rng.choice(len(q), p=q)                    # draft proposes x ~ q
    if rng.random() < min(1.0, p[x] / q[x]):       # accept: min(1, p/q)
        return x, True
    residual = np.maximum(p - q, 0.0)              # else: sample residual
    residual /= residual.sum()
    return rng.choice(len(p), p=residual), False


if __name__ == "__main__":
    print("1) average tokens per main-model pass")
    for depth in (1, 2, 3):
        for alpha in (0.5, 0.7, 0.85, 0.9):
            print(f"   D={depth} alpha={alpha:.2f}: simulated "
                  f"{tokens_per_step(alpha, depth):.3f}, "
                  f"analytic {expected_tokens(alpha, depth):.3f}")

    print("2) exactness of the acceptance rule")
    p = np.array([0.50, 0.25, 0.15, 0.10])         # main model
    q = np.array([0.10, 0.20, 0.30, 0.40])         # deliberately bad draft
    n = 100_000
    draws, acc = zip(*(speculative_sample(p, q) for _ in range(n)))
    freq = np.bincount(draws, minlength=len(p)) / n
    print("   target p   :", p)
    print("   empirical  :", np.round(freq, 3))
    print(f"   acceptance : {np.mean(acc):.3f} "
          f"(theory: sum(min(p, q)) = {np.minimum(p, q).sum():.3f})")
