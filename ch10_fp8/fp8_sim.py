"""FP8 (E4M3 / E5M2) emulation in NumPy: per-tensor vs fine-grained scaling.

Toy implementation for Chapter 10 of "Secrets of DeepSeek AI" (Listing 10.1).
We enumerate every value an 8-bit float can hold, round real numbers to the nearest one, and
compare two ways of choosing the scale factor:

  * per-tensor  - one scale for the whole matrix (classic FP8 recipe);
  * fine-grained - DeepSeek-V3 style: one scale per 1x128 tile of activations
                   and one per 128x128 block of weights.

Run:
    python fp8_sim.py
"""
import numpy as np


def fp8_grid(exp_bits, man_bits, bias, ieee_like):
    """All non-negative finite values of a tiny float format."""
    vals = [0.0]
    max_exp = 2 ** exp_bits - 1
    for m in range(1, 2 ** man_bits):                  # subnormals
        vals.append(m / 2 ** man_bits * 2.0 ** (1 - bias))
    top = max_exp - 1 if ieee_like else max_exp        # E5M2 reserves inf/NaN
    for e in range(1, top + 1):
        for m in range(2 ** man_bits):
            if not ieee_like and e == max_exp and m == 2 ** man_bits - 1:
                continue                               # E4M3: S.1111.111 is NaN
            vals.append((1 + m / 2 ** man_bits) * 2.0 ** (e - bias))
    return np.array(sorted(vals))


E4M3 = fp8_grid(4, 3, 7, ieee_like=False)      # max 448
E5M2 = fp8_grid(5, 2, 15, ieee_like=True)      # max 57344


def round_to_grid(x, grid):
    """Round to the nearest representable value (saturating at +-max)."""
    a = np.clip(np.abs(x), 0, grid[-1])
    idx = np.clip(np.searchsorted(grid, a), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    q = np.where(a - lo <= hi - a, lo, hi)
    return np.sign(x) * q


def quantize(x, grid, scale):
    """x ~ scale * fp8(x / scale); returns the dequantized approximation."""
    return round_to_grid(x / scale, grid) * scale


def per_tensor(x, grid):
    """One scale for the whole tensor, formula (10.3)."""
    s = np.max(np.abs(x)) / grid[-1]
    if s == 0:                                         # all-zero tensor
        s = 1.0
    return quantize(x, grid, s)


def per_tile(x, grid, tile=128):
    """Activations: one scale per 1 x `tile` segment, formula (10.4)."""
    out = np.empty_like(x)
    for j in range(0, x.shape[1], tile):
        blk = x[:, j:j + tile]
        s = np.max(np.abs(blk), axis=1, keepdims=True) / grid[-1]
        s[s == 0] = 1.0
        out[:, j:j + tile] = quantize(blk, grid, s)
    return out


def per_block(w, grid, block=128):
    """Weights: one scale per block x block square (128 x 128)."""
    out = np.empty_like(w)
    for i in range(0, w.shape[0], block):
        for j in range(0, w.shape[1], block):
            blk = w[i:i + block, j:j + block]
            s = np.max(np.abs(blk)) / grid[-1] or 1.0
            out[i:i + block, j:j + block] = quantize(blk, grid, s)
    return out


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


if __name__ == "__main__":
    print(f"E4M3: {len(E4M3)} non-negative values, max {E4M3[-1]:g}, "
          f"min subnormal {E4M3[1]:g}")
    print(f"E5M2: {len(E5M2)} non-negative values, max {E5M2[-1]:g}, "
          f"min subnormal {E5M2[1]:g}")

    rng = np.random.default_rng(0)
    tokens, d_in, d_out = 64, 1024, 256
    X = rng.standard_normal((tokens, d_in))
    X[:, rng.choice(d_in, 4, replace=False)] *= 200.0   # a few outlier channels
    W = rng.standard_normal((d_in, d_out)) * 0.02

    for name, grid in (("E4M3", E4M3), ("E5M2", E5M2)):
        xt, xf = per_tensor(X, grid), per_tile(X, grid)
        wt, wf = per_tensor(W, grid), per_block(W, grid)
        print(f"\n[{name}] activation error  per-tensor {rel_err(xt, X):.4f}"
              f" | 1x128 tiles {rel_err(xf, X):.4f}")
        print(f"[{name}] GEMM output error   per-tensor {rel_err(xt @ wt, X @ W):.4f}"
              f" | fine-grained {rel_err(xf @ wf, X @ W):.4f}")

    # how many activations collapse to zero under one global scale?
    zeros_t = np.mean(per_tensor(X, E4M3) == 0) * 100
    zeros_f = np.mean(per_tile(X, E4M3) == 0) * 100
    print(f"\nE4M3 values flushed to zero: per-tensor {zeros_t:.1f}% | tiles {zeros_f:.1f}%")
