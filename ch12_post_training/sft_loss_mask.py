"""Supervised fine-tuning loss with prompt masking and sample packing.

Teaching implementation in NumPy for Chapter 12 of "Secrets of DeepSeek AI".
Two ideas are shown:

1. The SFT loss is averaged only over response tokens: prompt tokens are
   masked out, so the model learns to answer, not to repeat questions.
2. When several short samples are packed into one training sequence, a
   block-diagonal causal mask keeps them isolated from each other.

Run:
    python sft_loss_mask.py
"""
import numpy as np

IGNORE = -100   # conventional "do not train on this position" label


def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def build_labels(prompt_ids, response_ids):
    """Concatenate prompt and response; labels are the next tokens, with
    every position that predicts a prompt token set to IGNORE."""
    tokens = np.array(prompt_ids + response_ids)
    labels = np.full(len(tokens), IGNORE)
    # position t predicts token t+1; train only where t+1 lies in the response
    for t in range(len(tokens) - 1):
        if t + 1 >= len(prompt_ids):
            labels[t] = tokens[t + 1]
    return tokens, labels


def sft_loss(logits, labels):
    """Mean cross-entropy over positions whose label is not IGNORE."""
    logp = log_softmax(logits, axis=-1)
    keep = labels != IGNORE
    picked = logp[np.arange(len(labels))[keep], labels[keep]]
    return -picked.mean(), int(keep.sum())


def packed_causal_mask(lengths):
    """Block-diagonal causal mask for samples packed back to back.
    True means "may attend"."""
    n = sum(lengths)
    mask = np.zeros((n, n), dtype=bool)
    start = 0
    for L in lengths:
        block = np.tril(np.ones((L, L), dtype=bool))
        mask[start:start + L, start:start + L] = block
        start += L
    return mask


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    vocab = 32
    prompt = [5, 9, 14, 3]          # e.g. "<user> how many ... <assistant>"
    response = [21, 7, 30, 2]       # the answer, ending with <eos>=2
    tokens, labels = build_labels(prompt, response)
    print("tokens:", tokens.tolist())
    print("labels:", labels.tolist())

    logits = rng.standard_normal((len(tokens), vocab))
    loss, n = sft_loss(logits, labels)
    print(f"masked SFT loss on {n} response positions: {loss:.3f}")
    print(f"(a random model should sit near ln(V) = {np.log(vocab):.3f})")

    # without masking the loss would also train on reproducing the prompt
    full = np.append(tokens[1:], IGNORE)
    loss_all, n_all = sft_loss(logits, full)
    print(f"unmasked loss over {n_all} positions: {loss_all:.3f}")

    m = packed_causal_mask([3, 2, 3])
    print("packed mask for samples of length 3, 2, 3 (1 = may attend):")
    print(m.astype(int))
