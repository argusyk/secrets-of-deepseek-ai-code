"""Multi-Token Prediction (MTP) loss in the style of DeepSeek-V3, in NumPy.

A toy implementation of sequential MTP modules (depth D, default D = 1) on
top of a toy "main model". It shows the two things that are easy to get
wrong:

  * how the targets are shifted (the main head predicts t[i+1], the depth-k
    module predicts t[i+k+1]);
  * how the module merges the previous hidden state with the embedding of
    the already-known token t[i+k] before its own transformer block.

The sequence is passed with D extra tokens at the end, so every one of the
T input positions has a target at every depth. Only the forward pass and the
losses are computed: no training. See Chapter 7 (Listing 7.1) of
"Secrets of DeepSeek AI"; the PyTorch counterpart is
`tinyseek.model.MTPModule`.

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


def init_weights(d=32, vocab=50, depth=1, seed=0):
    """Random weights: shared embedding/head, toy main model, D modules."""
    r = np.random.default_rng(seed)
    return {
        "emb": r.standard_normal((vocab, d)) * 0.1,     # shared embedding
        "W_out": r.standard_normal((d, vocab)) * 0.1,   # shared output head
        "W_main": r.standard_normal((d, d)) * 0.1,      # toy main model
        "M": [r.standard_normal((2 * d, d)) * 0.1 for _ in range(depth)],
        "W_blk": [r.standard_normal((d, d)) * 0.1 for _ in range(depth)],
    }


def mtp_losses(tokens, w, depth=1, lam=0.3):
    """Main next-token loss and MTP losses for one sequence.

    tokens: int array of length T + 1 + depth.
    Returns (L_main, [L_mtp_1, ..., L_mtp_D], L_total).
    """
    T = len(tokens) - 1 - depth           # positions used as inputs
    emb, W_out = w["emb"], w["W_out"]     # shared with the main model
    # main model (depth 0): input t[i], target t[i+1]
    h = toy_block(emb[tokens[:T]], w["W_main"])
    L_main = cross_entropy(rms_norm(h) @ W_out, tokens[1:T + 1])
    L_mtp = []
    for k in range(1, depth + 1):
        # [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t[i+k]))] -> projection M_k
        e = emb[tokens[k:k + T]]
        merged = np.concatenate([rms_norm(h), rms_norm(e)], axis=-1)
        h = toy_block(merged @ w["M"][k - 1], w["W_blk"][k - 1])
        # the shared output head predicts t[i+k+1]
        logits = rms_norm(h) @ W_out
        L_mtp.append(cross_entropy(logits, tokens[k + 1:k + 1 + T]))
    L_total = L_main + lam / depth * sum(L_mtp)
    return L_main, L_mtp, L_total


if __name__ == "__main__":
    vocab = 50
    seq = rng.integers(0, vocab, size=12)
    print("tokens:", seq)
    print("main head  predicts t[i+1]  for inputs t[i]")
    print("MTP depth 1 predicts t[i+2] using h_i and Emb(t[i+1])")
    w = init_weights(vocab=vocab, depth=1)
    for lam in (0.3, 0.1):
        L_main, L_mtp, L = mtp_losses(seq, w, depth=1, lam=lam)
        print(f"lambda={lam}: L_main={L_main:.4f}  "
              f"L_mtp={L_mtp[0]:.4f}  total={L:.4f}")
    # depth 2: every input position still has a target at both depths
    w2 = init_weights(vocab=vocab, depth=2)
    L_main, L_mtp, L = mtp_losses(seq, w2, depth=2, lam=0.3)
    print("depth 2: L_mtp =", [round(float(x), 4) for x in L_mtp])
    # an untrained model sits near ln(vocab) for every head
    print(f"reference ln(vocab) = {np.log(vocab):.4f}")
