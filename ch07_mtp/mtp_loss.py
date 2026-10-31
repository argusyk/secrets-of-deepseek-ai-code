"""Multi-Token Prediction (MTP) loss in the style of DeepSeek-V3, in NumPy.

A teaching implementation of one sequential MTP module of depth k = 1 on top
of a toy "main model". It shows the two things that are easy to get wrong:

  * how the targets are shifted (the main head predicts t[i+1], the depth-k
    module predicts t[i+k+1]);
  * how the module merges the previous hidden state with the embedding of the
    already-known token t[i+k] before its own transformer block.

Only the forward pass and the losses are computed: no training. See Chapter 7
of "Secrets of DeepSeek AI".

Run:
    python mtp_loss.py
"""
import numpy as np

rng = np.random.default_rng(7)


def rms_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def cross_entropy(logits, targets):
    """Mean negative log-likelihood of `targets` under `logits` (rows)."""
    lp = log_softmax(logits)
    return -lp[np.arange(len(targets)), targets].mean()


def toy_block(x, W):
    """Stand-in for a transformer block: residual + nonlinearity.
    (A real block has causal attention and an FFN/MoE.)"""
    return x + np.tanh(x @ W)


def mtp_losses(tokens, d=32, vocab=50, depth=1, lam=0.3, seed=0):
    """Compute the main next-token loss and the MTP loss for one sequence.

    tokens: int array of length T+1+depth (inputs plus the shifted targets).
    Returns (L_main, [L_mtp_k], L_total).
    """
    rng = np.random.default_rng(seed)               # same weights per call
    T = len(tokens) - 1 - depth                     # positions used as inputs
    emb = rng.standard_normal((vocab, d)) * 0.1      # shared embedding
    W_out = rng.standard_normal((d, vocab)) * 0.1    # shared output head
    W_main = rng.standard_normal((d, d)) * 0.1       # toy main model
    M = [rng.standard_normal((2 * d, d)) * 0.1 for _ in range(depth)]
    W_blk = [rng.standard_normal((d, d)) * 0.1 for _ in range(depth)]

    # main model: hidden states for positions 0..T-1 (depth 0)
    h = toy_block(emb[tokens[:T]], W_main)
    L_main = cross_entropy(rms_norm(h) @ W_out, tokens[1:T + 1])

    L_mtp = []
    for k in range(1, depth + 1):
        n = T - k + 1                                # positions with a target
        # merge: [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))] -> projection M_k
        merged = np.concatenate(
            [rms_norm(h[:n]), rms_norm(emb[tokens[k:k + n]])], axis=-1) @ M[k - 1]
        h = toy_block(merged, W_blk[k - 1])          # the module's own block
        logits = rms_norm(h) @ W_out                 # shared output head
        L_mtp.append(cross_entropy(logits, tokens[k + 1:k + 1 + n]))

    L_total = L_main + lam / depth * sum(L_mtp)
    return L_main, L_mtp, L_total


if __name__ == "__main__":
    vocab = 50
    seq = rng.integers(0, vocab, size=12)
    print("tokens:", seq)
    print("main head  predicts t[i+1]  for inputs t[i]")
    print("MTP depth 1 predicts t[i+2] using h_i and Emb(t[i+1])")
    for lam in (0.3, 0.1):
        L_main, L_mtp, L = mtp_losses(seq, vocab=vocab, depth=1, lam=lam)
        print(f"lambda={lam}: L_main={L_main:.4f}  L_mtp={L_mtp[0]:.4f}  total={L:.4f}")
    # an untrained model sits near ln(vocab) for every head
    print(f"reference ln(vocab) = {np.log(vocab):.4f}")
