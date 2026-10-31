"""Speculative decoding with an MTP draft: a toy simulation in NumPy.

Two experiments (see Chapter 7 of "Secrets of DeepSeek AI"):

1. Throughput. If the MTP module drafts one extra token and the main model
   accepts it with probability alpha, each forward step of the main model
   yields 1 + alpha tokens on average. With 85-90% acceptance this is
   1.85-1.9 tokens per step before overheads.

2. Exactness. The acceptance rule min(1, p/q) plus resampling from the
   normalized residual max(0, p - q) keeps the output distribution equal to
   the main model's distribution p, however poor the draft distribution q is.

Run:
    python spec_decode_toy.py
"""
import numpy as np

rng = np.random.default_rng(42)


def tokens_per_step(alpha, steps=200_000):
    """Monte Carlo: one guaranteed token + one drafted token accepted w.p. alpha."""
    accepted = rng.random(steps) < alpha
    return 1.0 + accepted.mean()


def speculative_sample(p, q):
    """Draw one token whose distribution is exactly p, using a draft from q."""
    x = rng.choice(len(q), p=q)                      # draft proposes x ~ q
    if rng.random() < min(1.0, p[x] / q[x]):         # accept w.p. min(1, p/q)
        return x, True
    residual = np.maximum(p - q, 0.0)                # otherwise resample
    residual /= residual.sum()
    return rng.choice(len(p), p=residual), False


if __name__ == "__main__":
    print("1) average tokens per main-model step with a depth-1 draft")
    for alpha in (0.5, 0.7, 0.85, 0.9):
        print(f"   alpha={alpha:.2f}: simulated {tokens_per_step(alpha):.3f}, "
              f"analytic {1 + alpha:.3f}")

    print("2) exactness of the acceptance rule")
    p = np.array([0.50, 0.25, 0.15, 0.10])           # main model
    q = np.array([0.10, 0.20, 0.30, 0.40])           # deliberately bad draft
    n = 100_000
    draws, acc = zip(*(speculative_sample(p, q) for _ in range(n)))
    freq = np.bincount(draws, minlength=len(p)) / n
    print("   target p   :", p)
    print("   empirical  :", np.round(freq, 3))
    print(f"   acceptance : {np.mean(acc):.3f} "
          f"(theory: sum(min(p, q)) = {np.minimum(p, q).sum():.3f})")
