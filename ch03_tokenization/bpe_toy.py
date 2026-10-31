"""A toy byte-pair-encoding (BPE) trainer and encoder.

Teaching code for Chapter 3 of "Secrets of DeepSeek AI". It learns merges on a
tiny corpus exactly as described in the chapter: count adjacent symbol pairs,
merge the most frequent one, repeat. Real tokenizers (including DeepSeek's
byte-level BPE with a ~128K vocabulary) work on bytes, use pre-tokenization
rules and are trained on terabytes; the core loop is the same.

Run:
    python bpe_toy.py
"""
from collections import Counter

END = "</w>"   # end-of-word marker so that "low" and "lower" stay distinct


def word_freqs(corpus):
    """Split text into words and represent each word as a tuple of symbols."""
    freqs = Counter(corpus.split())
    return {tuple(w) + (END,): c for w, c in freqs.items()}


def pair_counts(vocab):
    pairs = Counter()
    for symbols, c in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += c
    return pairs


def merge_pair(vocab, pair):
    a, b = pair
    merged = {}
    for symbols, c in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = c
    return merged


def train_bpe(corpus, n_merges):
    vocab = word_freqs(corpus)
    merges = []
    for _ in range(n_merges):
        pairs = pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_pair(vocab, best)
    return merges, vocab


def encode(word, merges):
    """Apply learned merges in order to a new word."""
    symbols = list(word) + [END]
    for a, b in merges:
        i, out = 0, []
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


if __name__ == "__main__":
    corpus = ("low low low low low lower lower newest newest newest "
              "newest newest newest widest widest widest")
    merges, vocab = train_bpe(corpus, n_merges=10)
    print("learned merges (in order):")
    for k, m in enumerate(merges, 1):
        print(f"  {k:2d}. {m[0]!r} + {m[1]!r}")
    print("\nfinal segmentation of the corpus words:")
    for symbols, c in vocab.items():
        print(f"  {c} x {' '.join(symbols)}")
    for w in ["lowest", "newer", "wider"]:
        print(f"\nunseen word {w!r} ->", encode(w, merges))
