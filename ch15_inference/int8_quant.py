"""Symmetric weight quantization: per-tensor vs per-channel vs group-wise.

Teaching example for Chapter 15 of "Secrets of DeepSeek AI". A weight matrix
with a few outlier channels (as real LLM layers have) is quantized to INT8 and
INT4 with different scale granularities; we report the relative error of the
weights and of a matrix-vector product, plus the memory footprint including
the scales. The point: one large value ruins a per-tensor scale, while finer
groups confine the damage.

Run:
    python int8_quant.py
"""
import numpy as np


def quantize(w, bits, axis=None, group=None):
    """Symmetric round-to-nearest quantization; returns (dequantized w, n_scales).

    axis=None, group=None -> one scale for the whole tensor
    axis=1                -> one scale per output row (per-channel)
    group=g               -> one scale per block of g consecutive inputs in a row
    """
    qmax = 2 ** (bits - 1) - 1
    if group is not None:
        rows, cols = w.shape
        wg = w.reshape(rows, cols // group, group)
        s = np.abs(wg).max(axis=-1, keepdims=True) / qmax
        s[s == 0] = 1.0
        q = np.clip(np.round(wg / s), -qmax, qmax)
        return (q * s).reshape(rows, cols), s.size
    if axis is None:
        s = np.abs(w).max() / qmax
        q = np.clip(np.round(w / s), -qmax, qmax)
        return q * s, 1
    s = np.abs(w).max(axis=axis, keepdims=True) / qmax
    q = np.clip(np.round(w / s), -qmax, qmax)
    return q * s, s.size


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 1024, 1024
    W = rng.standard_normal((rows, cols)) * 0.02
    # a handful of outlier input channels, 50x larger than the rest
    W[:, rng.choice(cols, 8, replace=False)] *= 50
    x = rng.standard_normal(cols)
    y = W @ x

    configs = [
        ("INT8 per-tensor", 8, None, None),
        ("INT8 per-channel", 8, 1, None),
        ("INT8 group-128", 8, None, 128),
        ("INT4 per-channel", 4, 1, None),
        ("INT4 group-128", 4, None, 128),
        ("INT4 group-32", 4, None, 32),
    ]
    base_bytes = W.size * 2  # BF16 reference
    print(f"{'scheme':18s} {'weight err':>10s} {'output err':>10s} {'memory':>8s}")
    for name, bits, axis, group in configs:
        Wq, n_scales = quantize(W, bits, axis, group)
        mem = W.size * bits / 8 + n_scales * 2   # payload + BF16 scales
        print(f"{name:18s} {rel_err(W, Wq):10.4f} {rel_err(y, Wq @ x):10.4f} "
              f"{mem / base_bytes:7.1%}")
