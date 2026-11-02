"""Local corpus, deduplication, packing and arithmetic tasks (Chapters 3, 12-14).

The TinySeek corpus is built offline, without any network access, from three
sources that imitate the mixture of a real pre-training corpus (Section 3.4):

* code - source files of the Python standard library of the running
  interpreter (so the exact corpus depends on the Python version);
* prose - Ukrainian and English texts shipped in ``tinyseek/data_texts``
  (prose of this textbook with formulas, code and markup removed);
* math - synthetic arithmetic problems with a column-wise chain of thought
  in the ``<think>...</think><answer>...</answer>`` format of Chapter 13.

Near-duplicate documents are removed with MinHash + LSH (Section 3.5), code
documents get the FIM transform (Chapter 17), and `PackedDataset` packs the
token stream into fixed-length training sequences.

CLI::

    python -m tinyseek.data build-corpus --out runs/data
    python -m tinyseek.data prepare --out runs/data --vocab-size 2048
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sysconfig
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .tokenizer import ByteBPETokenizer, fim_transform

TEXTS_DIR = Path(__file__).resolve().parent / "data_texts"
DEFAULT_DATA_DIR = Path("runs/data")


# ------------------------------------------------------------------ arithmetic
def make_arith_task(rng, digits: int = 2, ops: str = "+-") -> tuple[str, str]:
    """A random arithmetic problem: returns ``(prompt, answer)``.

    ``rng`` is a NumPy Generator or ``random.Random``. Operands have up to
    ``digits`` digits; subtraction never goes below zero.
    """
    hi = 10 ** digits
    if hasattr(rng, "integers"):
        a, b = int(rng.integers(0, hi)), int(rng.integers(0, hi))
        op = ops[int(rng.integers(0, len(ops)))]
    else:
        a, b, op = rng.randrange(hi), rng.randrange(hi), rng.choice(ops)
    if op == "-" and a < b:
        a, b = b, a
    result = a + b if op == "+" else a - b
    return f"Q: {a}{op}{b}=?\nA: ", str(result)


def parse_arith_prompt(prompt: str) -> tuple[int, str, int]:
    """Inverse of the prompt format: ``"Q: 47+38=?\\nA: "`` -> (47, '+', 38)."""
    expr = prompt.split("Q:", 1)[1].split("=", 1)[0].strip()
    for op in "+-":
        if op in expr:
            a, b = expr.split(op)
            return int(a), op, int(b)
    raise ValueError(f"not an arithmetic prompt: {prompt!r}")


def arith_reasoning(a: int, op: str, b: int) -> str:
    """Column-wise chain of thought, least significant digit first.

    ``47+38`` -> ``7+8=15 4+3+1=8``; ``52-38`` -> ``12-8=4 4-3=1``.
    """
    da, db = str(a)[::-1], str(b)[::-1]
    steps, carry = [], 0
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        if op == "+":
            s = x + y + carry
            steps.append(f"{x}+{y}+{carry}={s}" if carry else f"{x}+{y}={s}")
            carry = s // 10
        else:
            x -= carry
            carry = 1 if x < y else 0
            x += 10 * carry
            steps.append(f"{x}-{y}={x - y}")
    return " ".join(steps)


def arith_completion(prompt: str, answer: Optional[str] = None,
                     with_reasoning: bool = True) -> str:
    """Reference completion in the R1 format for an arithmetic prompt."""
    a, op, b = parse_arith_prompt(prompt)
    ans = answer if answer is not None else str(a + b if op == "+" else a - b)
    if not with_reasoning:
        return f"<answer>{ans}</answer>"
    return f"<think>{arith_reasoning(a, op, b)}</think><answer>{ans}</answer>"


# ------------------------------------------------------------------ dedup
def _shingle_hashes(text: str, k: int) -> np.ndarray:
    t = " ".join(text.lower().split())
    grams = {t[i:i + k] for i in range(max(1, len(t) - k + 1))}
    return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4)
                                    .digest(), "little") for g in grams],
                    dtype=np.uint64)


def minhash_signatures(docs: Sequence[str], num_perm: int = 64, k: int = 5,
                       seed: int = 0) -> np.ndarray:
    """MinHash signatures (Section 3.5, formula (3.3)): shape (n_docs, num_perm).

    Hash functions are h(x) = (a x + b) mod p with p = 2^32 - 5; all
    arithmetic fits in uint64.
    """
    p = np.uint64((1 << 32) - 5)
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(p), num_perm, dtype=np.uint64)
    b = rng.integers(0, int(p), num_perm, dtype=np.uint64)
    sig = np.empty((len(docs), num_perm), dtype=np.uint64)
    for i, d in enumerate(docs):
        x = _shingle_hashes(d, k)
        sig[i] = ((a[:, None] * x[None, :] + b[:, None]) % p).min(axis=1)
    return sig


def minhash_dedup(docs: Sequence[str], threshold: float = 0.8, num_perm: int = 64,
                  bands: int = 16, k: int = 5, seed: int = 0
                  ) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Near-duplicate removal with MinHash + LSH banding (formula (3.4)).

    Returns ``(keep, pairs)``: indices of the documents to keep (the first
    document of every near-duplicate group survives) and the detected pairs
    ``(i, j, estimated_jaccard)`` with estimate >= ``threshold``.
    """
    if not docs:
        return [], []
    sig = minhash_signatures(docs, num_perm, k, seed)
    rows = num_perm // bands
    buckets: dict[tuple, list[int]] = {}
    for i in range(len(docs)):
        for bnd in range(bands):
            key = (bnd, sig[i, bnd * rows:(bnd + 1) * rows].tobytes())
            buckets.setdefault(key, []).append(i)
    cand = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                cand.add((members[x], members[y]))
    pairs = []
    for i, j in sorted(cand):
        est = float(np.mean(sig[i] == sig[j]))
        if est >= threshold:
            pairs.append((i, j, est))
    parent = list(range(len(docs)))

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    keep = [i for i in range(len(docs)) if find(i) == i]
    return keep, pairs


# ------------------------------------------------------------------ corpus
def _split_of(text: str, val_fraction: float) -> str:
    h = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "little")
    return "val" if (h % 10_000) < val_fraction * 10_000 else "train"


def _paragraph_docs(text: str, target_chars: int = 1500) -> list[str]:
    """Group paragraphs into documents of roughly ``target_chars``."""
    docs, cur = [], []
    for para in text.split("\n\n"):
        cur.append(para.strip())
        if sum(len(c) for c in cur) >= target_chars:
            docs.append("\n\n".join(cur))
            cur = []
    if cur:
        docs.append("\n\n".join(cur))
    return docs


def _perturb(text: str, rng: random.Random, rate: float = 0.02) -> str:
    """A near-copy: ~``rate`` of the words replaced/dropped, whitespace changed."""
    words = text.split(" ")
    out = []
    for w in words:
        r = rng.random()
        if r < rate / 2:
            continue                                   # drop a word
        if r < rate:
            w = w[::-1]                                # garble a word
        out.append(w)
    return " ".join(out).replace("\n\n", "\n \n", 1)


def build_corpus(out_dir: str | Path = DEFAULT_DATA_DIR, code_mb: float = 1.0,
                 n_arith: int = 6000, arith_digits: Sequence[int] = (1, 2, 3),
                 fim_rate: float = 0.5, val_fraction: float = 0.05, seed: int = 0,
                 dedup_threshold: float = 0.8, dup_rate: float = 0.05,
                 verbose: bool = True) -> dict:
    """Assemble ``corpus.jsonl`` (one JSON document per line) in ``out_dir``.

    Each record has ``source`` (code / uk / en / math), ``split`` (train /
    val) and ``text``. Returns statistics, including the number of
    near-duplicates removed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict] = []

    # code: the standard library of this interpreter, deterministic order
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    budget = int(code_mb * 1_000_000)
    files = sorted(p for p in stdlib.glob("*.py") if not p.name.startswith("_"))
    code_docs, used = [], 0
    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # cut long modules into ~4 KB chunks at blank lines between defs
        chunk: list[str] = []
        for block in src.split("\n\n\n"):
            chunk.append(block)
            if sum(len(c) for c in chunk) > 4000:
                code_docs.append("\n\n\n".join(chunk))
                chunk = []
        if chunk:
            code_docs.append("\n\n\n".join(chunk))
        used += len(src.encode("utf-8"))
        if used >= budget:
            break
    docs = [("code", t) for t in code_docs]
    for lang in ("uk", "en"):                                   # prose
        f = TEXTS_DIR / f"{lang}_book.txt"
        if f.exists():
            docs += [(lang, t) for t in _paragraph_docs(f.read_text(encoding="utf-8"))]

    # web corpora are full of near-copies (mirrors, reposts, boilerplate);
    # we imitate them by re-inserting a share of documents with small edits
    n_orig = len(docs)
    for i in sorted(rng.sample(range(n_orig), int(dup_rate * n_orig))):
        docs.append((docs[i][0], _perturb(docs[i][1], rng)))
    keep, pairs = minhash_dedup([t for _, t in docs], threshold=dedup_threshold, seed=seed)
    n_dupes = len(docs) - len(keep)
    kept = set(keep)
    injected_found = sum(1 for i in range(n_orig, len(docs)) if i not in kept)
    for i in keep:
        records.append({"source": docs[i][0], "text": docs[i][1]})

    # synthetic arithmetic with chain of thought (DeepSeekMath-style data)
    nrng = np.random.default_rng(seed)
    for _ in range(n_arith // 8):
        lines = []
        for _ in range(8):
            prompt, ans = make_arith_task(nrng, int(nrng.choice(list(arith_digits))))
            lines.append(prompt + arith_completion(prompt, ans))
        records.append({"source": "math", "text": "\n".join(lines)})

    for r in records:
        r["split"] = _split_of(r["text"], val_fraction)
        if r["source"] == "code" and r["split"] == "train":
            r["text"] = fim_transform(r["text"], fim_rate, rng)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"documents": len(records), "near_duplicates_removed": n_dupes,
             "injected_duplicates": len(docs) - n_orig,
             "injected_found": injected_found,
             "bytes": {s: sum(len(r["text"].encode()) for r in records
                              if r["source"] == s) for s in ("code", "uk", "en", "math")}}
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    if verbose:
        print(f"corpus: {stats['documents']} documents, bytes {stats['bytes']}, "
              f"near-duplicates removed: {n_dupes} "
              f"(injected {len(docs) - n_orig}, caught {injected_found})")
    return stats


def load_corpus(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[dict]:
    path = Path(data_dir) / "corpus.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ packing
class PackedDataset:
    """A token stream packed into sequences of ``seq_len + 1`` tokens.

    Documents are joined with ``<|eos|>`` and the stream is cut into
    non-overlapping blocks (sample packing, Section 12.2). A batch returns
    inputs ``x = block[:-1]`` and next-token targets ``y = block[1:]``.
    """

    def __init__(self, tokens: Sequence[int] | np.ndarray | torch.Tensor,
                 seq_len: int) -> None:
        t = torch.as_tensor(np.asarray(tokens, dtype=np.int64))
        self.seq_len = seq_len
        n_blocks = (len(t) - 1) // seq_len
        if n_blocks < 1:
            raise ValueError("not enough tokens for a single sequence")
        self.tokens = t
        self.n_blocks = n_blocks

    @classmethod
    def from_texts(cls, texts: Sequence[str], tokenizer: ByteBPETokenizer,
                   seq_len: int) -> "PackedDataset":
        ids: list[int] = []
        for t in texts:
            ids.extend(tokenizer.encode(t))
            ids.append(tokenizer.eos_id)
        return cls(ids, seq_len)

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = i * self.seq_len
        block = self.tokens[s:s + self.seq_len + 1]
        return block[:-1], block[1:]

    def get_batch(self, batch_size: int, generator: Optional[torch.Generator] = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """A random batch of blocks (sampling with replacement)."""
        idx = torch.randint(0, self.n_blocks, (batch_size,), generator=generator)
        xs, ys = zip(*(self[int(i)] for i in idx))
        return torch.stack(xs), torch.stack(ys)

    def iter_batches(self, batch_size: int, max_batches: Optional[int] = None):
        """Deterministic pass over the blocks in order (for evaluation)."""
        n = self.n_blocks // batch_size
        if max_batches is not None:
            n = min(n, max_batches)
        for b in range(max(1, n)):
            items = [self[(b * batch_size + j) % self.n_blocks] for j in range(batch_size)]
            xs, ys = zip(*items)
            yield torch.stack(xs), torch.stack(ys)


def prepare_data(data_dir: str | Path = DEFAULT_DATA_DIR, vocab_size: int = 2048,
                 seed: int = 0, verbose: bool = True) -> tuple[ByteBPETokenizer, dict]:
    """Make sure the corpus, a tokenizer and the encoded splits exist (cached).

    Returns the tokenizer and ``{"train": np.ndarray, "val": np.ndarray,
    "val_by_source": {source: np.ndarray}}``.
    """
    d = Path(data_dir)
    if not (d / "corpus.jsonl").exists():
        build_corpus(d, seed=seed, verbose=verbose)
    tok_path = d / f"tokenizer_{vocab_size}.json"
    corpus = None
    if tok_path.exists():
        tok = ByteBPETokenizer.load(tok_path)
    else:
        corpus = load_corpus(d)
        if verbose:
            print(f"training BPE tokenizer, vocab {vocab_size} ...")
        tok = ByteBPETokenizer().train([r["text"] for r in corpus if r["split"] == "train"],
                                       vocab_size)
        tok.save(tok_path)
    npz = d / f"tokens_{vocab_size}.npz"
    if npz.exists():
        z = np.load(npz)
        arrays = {k: z[k] for k in z.files}
    else:
        corpus = corpus or load_corpus(d)
        arrays = {}
        for split in ("train", "val"):
            ids: list[int] = []
            for r in corpus:
                if r["split"] == split:
                    ids.extend(tok.encode(r["text"]))
                    ids.append(tok.eos_id)
            arrays[split] = np.array(ids, dtype=np.int32)
        for src in ("code", "uk", "en", "math"):
            ids = []
            for r in corpus:
                if r["split"] == "val" and r["source"] == src:
                    ids.extend(tok.encode(r["text"]))
                    ids.append(tok.eos_id)
            arrays[f"val_{src}"] = np.array(ids, dtype=np.int32)
        np.savez_compressed(npz, **arrays)
        if verbose:
            print(f"encoded: train {len(arrays['train']):,} tokens, "
                  f"val {len(arrays['val']):,} tokens")
    data = {"train": arrays["train"], "val": arrays["val"],
            "val_by_source": {k[4:]: v for k, v in arrays.items()
                              if k.startswith("val_") and len(v)}}
    return tok, data


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the offline TinySeek corpus "
                                "(Chapter 3).")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-corpus", help="assemble corpus.jsonl")
    b.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    b.add_argument("--code-mb", type=float, default=1.0)
    b.add_argument("--n-arith", type=int, default=6000)
    b.add_argument("--fim-rate", type=float, default=0.5)
    b.add_argument("--dup-rate", type=float, default=0.05,
                   help="share of documents re-inserted as near-copies")
    b.add_argument("--seed", type=int, default=0)
    pr = sub.add_parser("prepare", help="corpus + tokenizer + encoded tokens")
    pr.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    pr.add_argument("--vocab-size", type=int, default=2048)
    pr.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "build-corpus":
        build_corpus(args.out, code_mb=args.code_mb, n_arith=args.n_arith,
                     fim_rate=args.fim_rate, dup_rate=args.dup_rate, seed=args.seed)
    else:
        prepare_data(args.out, args.vocab_size, args.seed)


if __name__ == "__main__":
    main()
