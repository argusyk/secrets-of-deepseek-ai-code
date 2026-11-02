"""Why DeepSeek-V3 promotes FP8 GEMM partial sums to FP32 every 128 elements.

Toy implementation for Chapter 10 of "Secrets of DeepSeek AI" (Table 10.2).
Tensor cores add FP8 products into an accumulator that keeps only a limited
number of mantissa bits. We emulate such an accumulator by rounding the running sum to `m` bits
after every addition, then compare it with the remedy used in DeepSeek-V3:
accumulate an interval of N_C = 128 products in limited precision, then add
that partial sum into a full FP32 accumulator.

Run:
    python accumulation_demo.py
"""
import numpy as np


def round_bits(x, m):
    """Round x to m mantissa bits (same exponent range as float64)."""
    if x == 0.0:
        return 0.0
    mant, exp = np.frexp(x)                    # x = mant * 2**exp, 0.5 <= |mant| < 1
    scale = 2.0 ** (m + 1)
    return np.ldexp(np.round(mant * scale) / scale, exp)


def dot_limited(a, b, m):
    acc = 0.0
    for p in a * b:
        acc = round_bits(acc + p, m)
    return acc


def dot_promoted(a, b, m, interval=128):
    total = np.float32(0.0)                    # full-precision accumulator
    for start in range(0, len(a), interval):
        part = dot_limited(a[start:start + interval], b[start:start + interval], m)
        total = np.float32(total + np.float32(part))
    return float(total)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    m = 13                                     # roughly a 14-bit accumulator
    print(f"accumulator mantissa bits: {m}")
    print(f"{'K':>6} {'limited':>12} {'promoted/128':>14}")
    for K in (256, 1024, 4096, 16384):
        errs_l, errs_p = [], []
        for _ in range(20):
            # positive-mean inputs: the running sum grows, so its ulp grows too
            a = np.abs(rng.standard_normal(K)) + 0.1
            b = np.abs(rng.standard_normal(K)) + 0.1
            exact = float(np.dot(a, b))
            errs_l.append(abs(dot_limited(a, b, m) - exact) / exact)
            errs_p.append(abs(dot_promoted(a, b, m) - exact) / exact)
        print(f"{K:>6} {np.max(errs_l):>12.2e} {np.max(errs_p):>14.2e}")
    print("\nThe limited accumulator's error grows with K (for large K the sum")
    print("stagnates: new products fall below half an ulp of the running sum);")
    print("with promotion every 128 products the error does not grow.")
