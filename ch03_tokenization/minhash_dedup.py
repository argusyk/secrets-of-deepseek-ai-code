"""Near-duplicate detection with MinHash and locality-sensitive hashing (LSH).

Teaching code for Chapter 3 of "Secrets of DeepSeek AI". Large pre-training
pipelines remove near-duplicate documents before training; MinHash estimates
the Jaccard similarity of two documents' shingle sets from short signatures,
and LSH banding finds candidate pairs without comparing every pair.

Run:
    python minhash_dedup.py
"""
import hashlib
from itertools import combinations

import numpy as np

P = (1 << 61) - 1          # a large Mersenne prime for universal hashing


def shingles(text, k=5):
    """Character k-grams of a whitespace-normalized lowercase text."""
    t = " ".join(text.lower().split())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, n_perm=128, seed=0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, P - 1, n_perm, dtype=np.uint64).astype(object)
        self.b = rng.integers(0, P - 1, n_perm, dtype=np.uint64).astype(object)

    def signature(self, shingle_set):
        hashes = [h64(s) for s in shingle_set]
        sig = []
        for a, b in zip(self.a, self.b):
            sig.append(min((a * x + b) % P for x in hashes))
        return np.array(sig, dtype=object)


def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)


def lsh_candidates(signatures, bands=32):
    """Docs whose signatures agree on all rows of at least one band."""
    n_perm = len(next(iter(signatures.values())))
    rows = n_perm // bands
    buckets = {}
    for doc, sig in signatures.items():
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets.setdefault(key, set()).add(doc)
    cand = set()
    for docs in buckets.values():
        for pair in combinations(sorted(docs), 2):
            cand.add(pair)
    return cand


if __name__ == "__main__":
    docs = {
        "a": "DeepSeek-V3 is a Mixture-of-Experts model with 671B total parameters.",
        "b": "DeepSeek-V3 is a Mixture-of-Experts model with 671B total parameters!",
        "c": "DeepSeek V3 is a mixture of experts model with 671B parameters in total.",
        "d": "Byte-pair encoding merges the most frequent pair of adjacent symbols.",
    }
    sh = {k: shingles(v) for k, v in docs.items()}
    mh = MinHasher(n_perm=128)
    sigs = {k: mh.signature(v) for k, v in sh.items()}

    print("pair  true-Jaccard  MinHash-estimate")
    for x, y in combinations(docs, 2):
        est = float(np.mean(sigs[x] == sigs[y]))
        print(f"{x}-{y}   {jaccard(sh[x], sh[y]):.2f}          {est:.2f}")

    cand = lsh_candidates(sigs, bands=32)
    print("\nLSH candidate near-duplicate pairs:", sorted(cand))
    print("keep one document from each candidate group before training.")
