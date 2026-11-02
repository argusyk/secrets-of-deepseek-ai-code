"""How a language model picks the next token.

Greedy decoding, temperature, top-k and top-p (nucleus) sampling over one
vector of logits. A teaching implementation in NumPy for Chapter 2 of
"Secrets of DeepSeek AI"; the same parameters (temperature, top_k, top_p)
appear in tinyseek.generate.sample and in the DeepSeek API.

Run:
    python sampling.py
"""
import numpy as np


def softmax(z):
    z = z - np.max(z)                         # numerical stability
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits, temperature):
    """Divide the logits by tau > 0; small tau sharpens, large tau flattens."""
    return logits / temperature


def top_k_filter(logits, k):
    """Keep the k largest logits, set the rest to -inf."""
    if k is None or k >= logits.size:
        return logits
    kth = np.sort(logits)[-k]
    return np.where(logits < kth, -np.inf, logits)


def top_p_filter(logits, p):
    """Keep the smallest set of tokens whose total probability >= p."""
    if p is None or p >= 1.0:
        return logits
    probs = softmax(logits)
    order = np.argsort(-probs)                # most probable first
    cum = np.cumsum(probs[order])
    n_keep = int(np.searchsorted(cum, p)) + 1
    out = np.full_like(logits, -np.inf)
    out[order[:n_keep]] = logits[order[:n_keep]]
    return out


def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """One decoding step; temperature=0 means greedy decoding (argmax)."""
    if temperature == 0:
        return int(np.argmax(logits))
    z = apply_temperature(np.asarray(logits, dtype=float), temperature)
    z = top_k_filter(z, top_k)
    z = top_p_filter(z, top_p)
    return int(rng.choice(z.size, p=softmax(z)))


# A toy distribution: continuations of "The key is in the ..."
WORDS = ["drawer", "pocket", "door", "bag", "lock", "grass", "bank", "cloud"]
LOGITS = np.array([3.0, 2.6, 2.2, 1.9, 1.2, 0.3, -0.5, -1.5])


if __name__ == "__main__":
    print("Probabilities at different temperatures:")
    print(f"{'token':>8s}" + "".join(f"  tau={t:<4}" for t in (0.5, 1.0, 1.5)))
    table = {t: softmax(apply_temperature(LOGITS, t)) for t in (0.5, 1.0, 1.5)}
    for i, w in enumerate(WORDS):
        print(f"{w:>8s}" + "".join(f"  {table[t][i]:8.3f}" for t in table))

    p = softmax(LOGITS)
    ent = -np.sum(p * np.log(p))
    print(f"\nentropy at tau=1: {ent:.3f} nats; perplexity of this step:"
          f" {np.exp(ent):.2f} (out of {len(WORDS)} words)")

    kept = np.isfinite(top_k_filter(LOGITS, 3))
    print("top-k (k=3) keeps:", [w for w, k in zip(WORDS, kept) if k])
    for top_p in (0.5, 0.8):
        kept = np.isfinite(top_p_filter(LOGITS, top_p))
        print(f"top-p (p={top_p}) keeps:",
              [w for w, k in zip(WORDS, kept) if k])

    rng = np.random.default_rng(0)
    settings = [("greedy", dict(temperature=0)),
                ("tau=1.0", dict(temperature=1.0)),
                ("tau=0.6, top-p=0.95", dict(temperature=0.6, top_p=0.95)),
                ("tau=1.5, top-k=3", dict(temperature=1.5, top_k=3))]
    n = 10_000
    print(f"\nEmpirical frequencies over {n} draws:")
    for name, kw in settings:
        counts = np.bincount([sample_next(LOGITS, rng, **kw)
                              for _ in range(n)], minlength=len(WORDS))
        top = ", ".join(f"{WORDS[i]} {counts[i] / n:.2f}"
                        for i in np.argsort(-counts)[:4] if counts[i])
        distinct = int(np.count_nonzero(counts))
        print(f"  {name:22s} {distinct} distinct words; {top}")
