"""Multi-Head Latent Attention (MLA) in NumPy: naive vs. absorbed form.

Teaching implementation for Chapter 4 of "Secrets of DeepSeek AI".
It follows the MLA equations of DeepSeek-V2/V3:

    c_KV  = W_DKV h              (latent shared by keys and values; cached)
    k_C   = W_UK c_KV            (per-head "content" keys)
    v_C   = W_UV c_KV            (per-head values)
    k_R   = RoPE(W_KR h)         (decoupled positional key, shared by heads; cached)
    c_Q   = W_DQ h               (compressed query)
    q_C   = W_UQ c_Q,  q_R = RoPE(W_QR c_Q)
    score = (q_C . k_C + q_R . k_R) / sqrt(d_h + d_hR)

The "naive" form materialises full keys and values for every head.
The "absorbed" form never does: W_UK is folded into the query and
W_UV into the output projection, so attention runs directly over the
cached latent c_KV. Both forms must give the same output.

Run:
    python mla_numpy.py
"""
import numpy as np


def rope(x, pos, base=10000.0):
    """Rotary position embedding for the last dimension (must be even)."""
    d = x.shape[-1]
    half = d // 2
    freqs = base ** (-np.arange(half) / half)
    ang = pos[:, None] * freqs[None, :]               # (seq, half)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_params(rng, d=64, n_h=4, d_h=16, d_c=32, d_cq=48, d_hR=8):
    s = 0.2
    return dict(
        n_h=n_h, d_h=d_h, d_c=d_c, d_hR=d_hR,
        W_DKV=rng.standard_normal((d_c, d)) * s,
        W_UK=rng.standard_normal((n_h, d_h, d_c)) * s,
        W_UV=rng.standard_normal((n_h, d_h, d_c)) * s,
        W_KR=rng.standard_normal((d_hR, d)) * s,
        W_DQ=rng.standard_normal((d_cq, d)) * s,
        W_UQ=rng.standard_normal((n_h, d_h, d_cq)) * s,
        W_QR=rng.standard_normal((n_h, d_hR, d_cq)) * s,
        W_O=rng.standard_normal((d, n_h, d_h)) * s,
    )


def causal_mask(seq):
    return np.triu(np.ones((seq, seq), dtype=bool), k=1)


def mla_naive(H, P):
    """Materialise full per-head keys/values (what a training kernel sees)."""
    seq = H.shape[0]
    pos = np.arange(seq, dtype=float)
    c_kv = H @ P["W_DKV"].T                           # (seq, d_c)
    k_R = rope(H @ P["W_KR"].T, pos)                  # (seq, d_hR)
    c_q = H @ P["W_DQ"].T                             # (seq, d_cq)
    scale = 1.0 / np.sqrt(P["d_h"] + P["d_hR"])
    out = np.zeros_like(H)
    for i in range(P["n_h"]):
        k_C = c_kv @ P["W_UK"][i].T                   # (seq, d_h)
        v_C = c_kv @ P["W_UV"][i].T                   # (seq, d_h)
        q_C = c_q @ P["W_UQ"][i].T                    # (seq, d_h)
        q_R = rope(c_q @ P["W_QR"][i].T, pos)         # (seq, d_hR)
        k = np.concatenate([k_C, k_R], axis=-1)
        q = np.concatenate([q_C, q_R], axis=-1)
        s = q @ k.T * scale
        s = np.where(causal_mask(seq), -np.inf, s)
        o = softmax(s) @ v_C                          # (seq, d_h)
        out += o @ P["W_O"][:, i, :].T                # (seq, d)
    return out


def mla_absorbed(H, P):
    """Attend over the cached latent only; W_UK and W_UV are absorbed."""
    seq = H.shape[0]
    pos = np.arange(seq, dtype=float)
    cache_c = H @ P["W_DKV"].T                        # (seq, d_c)   <- KV cache
    cache_kR = rope(H @ P["W_KR"].T, pos)             # (seq, d_hR)  <- KV cache
    c_q = H @ P["W_DQ"].T
    scale = 1.0 / np.sqrt(P["d_h"] + P["d_hR"])
    out = np.zeros_like(H)
    for i in range(P["n_h"]):
        # fold W_UK into the query: q_C . (W_UK c) = (W_UK^T q_C) . c
        W_q_absorbed = P["W_UK"][i].T @ P["W_UQ"][i]  # (d_c, d_cq), precomputable
        q_lat = c_q @ W_q_absorbed.T                  # (seq, d_c)
        q_R = rope(c_q @ P["W_QR"][i].T, pos)
        s = (q_lat @ cache_c.T + q_R @ cache_kR.T) * scale
        s = np.where(causal_mask(seq), -np.inf, s)
        o_lat = softmax(s) @ cache_c                  # (seq, d_c): mix latents
        # fold W_UV into the output projection: W_O,i W_UV,i
        W_o_absorbed = P["W_O"][:, i, :] @ P["W_UV"][i]  # (d, d_c), precomputable
        out += o_lat @ W_o_absorbed.T
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    P = make_params(rng)
    H = rng.standard_normal((7, 64))
    y_naive = mla_naive(H, P)
    y_abs = mla_absorbed(H, P)
    err = np.max(np.abs(y_naive - y_abs))
    print("output shape      :", y_abs.shape)
    print("max |naive-absorbed| = %.2e" % err)
    assert np.allclose(y_naive, y_abs, atol=1e-10), "forms differ!"
    print("OK: absorbed MLA equals naive MLA")
    per_tok_mla = P["d_c"] + P["d_hR"]
    per_tok_mha = 2 * P["n_h"] * P["d_h"]
    print("cache per token per layer: MLA %d vs MHA %d elements (x%.1f smaller)"
          % (per_tok_mla, per_tok_mha, per_tok_mha / per_tok_mla))
