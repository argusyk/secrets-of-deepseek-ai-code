"""Scaled dot-product attention and a minimal multi-head wrapper.

A teaching implementation in NumPy: no autograd, no training, just the forward
pass so you can watch the shapes and the causal mask. See Chapter 2 of
"Secrets of DeepSeek AI".

Run:
    python scaled_dot_product_attention.py
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)      # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, causal=True):
    """Q, K, V: (seq, d_k). Returns (seq, d_v) and the attention matrix.

    scores = Q K^T / sqrt(d_k); optional causal mask hides future positions;
    weights = softmax(scores); output = weights @ V.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                  # (seq, seq)
    if causal:
        seq = scores.shape[0]
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)     # block attending ahead
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """X: (seq, d_model). W*: projection matrices. Splits into n_heads,
    runs attention per head, concatenates, and projects back."""
    seq, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                 # (seq, d_model)
    heads = []
    for h in range(n_heads):
        s = slice(h * d_head, (h + 1) * d_head)
        out, _ = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s], causal)
        heads.append(out)
    concat = np.concatenate(heads, axis=-1)          # (seq, d_model)
    return concat @ Wo


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq, d_model, n_heads = 5, 16, 4
    X = rng.standard_normal((seq, d_model))
    Wq, Wk, Wv, Wo = (rng.standard_normal((d_model, d_model)) for _ in range(4))
    Y = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads)
    print("input :", X.shape)
    print("output:", Y.shape)

    _, W = scaled_dot_product_attention(X[:, :4], X[:, :4], X[:, :4])
    print("row sums of the causal attention matrix (each ~1.0):")
    print(np.round(W.sum(axis=-1), 3))
    print("upper triangle is zero (no attending to the future):")
    print(np.round(W, 2))
