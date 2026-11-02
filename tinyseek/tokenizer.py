"""Byte-level BPE tokenizer and the FIM transform (Chapters 3 and 17).

`ByteBPETokenizer` is a compact version of the tokenizers used by DeepSeek:

* the base alphabet is the 256 byte values, so any UTF-8 text (Ukrainian,
  English, Python code) can be encoded without an "unknown" token;
* text is first split into pre-tokens by a regular expression (words with a
  leading space, single digits, runs of punctuation, whitespace); merges never
  cross a pre-token boundary. Digits are split one by one, as many LLM
  tokenizers do, which keeps arithmetic learnable for a tiny model;
* training repeatedly merges the most frequent adjacent pair (Section 3.2),
  with an incremental pair index and a lazy heap so that ~2K merges on a
  megabyte-sized corpus take seconds.

Special tokens: ``<|bos|>``, ``<|eos|>``, ``<|pad|>`` and the three FIM markers
``<|fim_begin|>``, ``<|fim_hole|>``, ``<|fim_end|>``. `fim_transform` rewrites
a document into the prefix-suffix-middle (PSM) order of DeepSeek-Coder.

CLI::

    python -m tinyseek.tokenizer train --corpus runs/data/corpus.jsonl \
        --vocab-size 2048 --out runs/data/tokenizer.json
    python -m tinyseek.tokenizer encode --tokenizer runs/data/tokenizer.json \
        --text "Привіт, DeepSeek!"
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>",
                  "<|fim_begin|>", "<|fim_hole|>", "<|fim_end|>"]

# letters/underscore words with an optional leading space | one digit |
# punctuation runs | whitespace (the last space is left for the next word)
PRETOKEN_RE = re.compile(r" ?[^\W\d]+|\d| ?[^\s\w]+|\s+(?!\S)|\s+")
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + ")")


def pretokenize(text: str) -> list[str]:
    """Split text into pre-tokens; ``"".join(pretokenize(t)) == t``."""
    return PRETOKEN_RE.findall(text)


class ByteBPETokenizer:
    """Byte-level BPE with special tokens (ids: 256 bytes, specials, merges)."""

    def __init__(self, merges: Optional[list[tuple[int, int]]] = None) -> None:
        self.special_tokens = list(SPECIAL_TOKENS)
        self.special_ids = {s: 256 + i for i, s in enumerate(self.special_tokens)}
        self.merges: list[tuple[int, int]] = []
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for s, i in self.special_ids.items():
            self.vocab[i] = s.encode("utf-8")
        self._cache: dict[str, list[int]] = {}
        for pair in merges or []:
            self._add_merge(tuple(pair))

    # ----------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.special_tokens) + len(self.merges)

    @property
    def bos_id(self) -> int:
        return self.special_ids["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_ids["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_ids["<|pad|>"]

    def _add_merge(self, pair: tuple[int, int]) -> int:
        new_id = 256 + len(self.special_tokens) + len(self.merges)
        self.merges.append(pair)
        self.ranks[pair] = len(self.merges) - 1
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return new_id

    # -------------------------------------------------------------- training
    def train(self, texts: Iterable[str], vocab_size: int = 2048,
              min_frequency: int = 2, verbose: bool = False) -> "ByteBPETokenizer":
        """Learn merges until the vocabulary has ``vocab_size`` entries.

        Ties between equally frequent pairs are broken by the smaller pair of
        ids, so training is fully deterministic.
        """
        self.__init__()
        counts: Counter[str] = Counter()
        for t in texts:
            for part in _SPECIAL_RE.split(t):
                if part and part not in self.special_ids:
                    counts.update(pretokenize(part))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())
        pair_count: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for a, b in zip(w, w[1:]):
                pair_count[(a, b)] += f
                where[(a, b)].add(wi)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)
        n_merges = vocab_size - 256 - len(self.special_tokens)
        while len(self.merges) < n_merges and heap:
            negc, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -negc:      # stale heap entry
                continue
            if -negc < min_frequency:
                break
            new_id = self._add_merge(pair)
            changed: dict[tuple[int, int], int] = {}
            for wi in list(where.pop(pair, ())):
                w, f = words[wi], freqs[wi]
                for a, b in zip(w, w[1:]):            # remove old pairs
                    pair_count[(a, b)] -= f
                    changed[(a, b)] = 1
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == pair[0] and w[i + 1] == pair[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for a, b in zip(out, out[1:]):        # add new pairs
                    pair_count[(a, b)] += f
                    where[(a, b)].add(wi)
                    changed[(a, b)] = 1
            pair_count.pop(pair, None)
            for p in changed:
                c = pair_count.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_count.pop(p, None)
            if verbose and len(self.merges) % 500 == 0:
                print(f"  merges: {len(self.merges)}")
        self._cache.clear()
        return self

    # -------------------------------------------------------------- encoding
    def _encode_word(self, word: str) -> list[int]:
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            best, best_rank = None, None
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            pair = (ids[best], ids[best + 1])
            new_id = 256 + len(self.special_tokens) + best_rank
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False,
               allow_special: bool = True) -> list[int]:
        """Text -> token ids. Special-token strings are recognized when
        ``allow_special`` is true (the default), otherwise encoded as bytes."""
        ids = [self.bos_id] if add_bos else []
        parts = _SPECIAL_RE.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_ids:
                ids.append(self.special_ids[part])
                continue
            for w in pretokenize(part):
                ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        """Token ids -> text (invalid UTF-8 sequences become U+FFFD)."""
        out = bytearray()
        for i in ids:
            i = int(i)
            if skip_special and 256 <= i < 256 + len(self.special_tokens):
                continue
            out += self.vocab[i]
        return out.decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Readable form of one token (for printing segmentations)."""
        return self.vocab[int(i)].decode("utf-8", errors="replace")

    # --------------------------------------------------------------- metrics
    def fertility(self, text: str) -> float:
        """Tokens per whitespace-separated word (Section 3.3); lower is better."""
        n_words = max(1, len(text.split()))
        return len(self.encode(text, allow_special=False)) / n_words

    def bytes_per_token(self, text: str) -> float:
        """Average UTF-8 bytes covered by one token (compression ratio)."""
        n = max(1, len(self.encode(text, allow_special=False)))
        return len(text.encode("utf-8")) / n

    # ------------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        data = {"type": "ByteBPETokenizer", "special_tokens": self.special_tokens,
                "merges": [list(m) for m in self.merges]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])


def fim_transform(text: str, rate: float, rng) -> str:
    """Fill-in-the-middle in PSM order (Chapter 17, DeepSeek-Coder).

    With probability ``rate`` the document is cut at two random character
    positions into prefix, middle and suffix and rewritten as
    ``<|fim_begin|>prefix<|fim_hole|>suffix<|fim_end|>middle``; otherwise it is
    returned unchanged. ``rng`` is a ``random.Random`` or a NumPy Generator.
    """
    if rng.random() >= rate or len(text) < 2:
        return text
    n = len(text)
    if hasattr(rng, "integers"):
        i, j = sorted(int(v) for v in rng.integers(0, n + 1, size=2))
    else:
        i, j = sorted((rng.randint(0, n), rng.randint(0, n)))
    prefix, middle, suffix = text[:i], text[i:j], text[j:]
    return f"<|fim_begin|>{prefix}<|fim_hole|>{suffix}<|fim_end|>{middle}"


def fim_inverse(text: str) -> str:
    """Undo `fim_transform` (useful for tests and for reading FIM samples)."""
    if not text.startswith("<|fim_begin|>"):
        return text
    body = text[len("<|fim_begin|>"):]
    prefix, rest = body.split("<|fim_hole|>", 1)
    suffix, middle = rest.split("<|fim_end|>", 1)
    return prefix + middle + suffix


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train or apply the TinySeek byte-level "
                                "BPE tokenizer (Chapter 3).")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train merges on a corpus.jsonl")
    t.add_argument("--corpus", required=True)
    t.add_argument("--vocab-size", type=int, default=2048)
    t.add_argument("--out", required=True)
    e = sub.add_parser("encode", help="show the segmentation of a text")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--text", required=True)
    args = p.parse_args(argv)
    if args.cmd == "train":
        texts = [json.loads(line)["text"] for line in
                 Path(args.corpus).read_text(encoding="utf-8").splitlines()]
        tok = ByteBPETokenizer().train(texts, args.vocab_size, verbose=True)
        tok.save(args.out)
        print(f"vocabulary {tok.vocab_size} -> {args.out}")
    else:
        tok = ByteBPETokenizer.load(args.tokenizer)
        ids = tok.encode(args.text)
        print(ids)
        print(" | ".join(tok.token_str(i) for i in ids))
        print(f"tokens={len(ids)} fertility={tok.fertility(args.text):.2f}")


if __name__ == "__main__":
    main()
