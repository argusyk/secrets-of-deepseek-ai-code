"""Byte-level BPE, fertility and FIM (Chapters 3 and 17)."""
import random

import numpy as np

from tinyseek.tokenizer import (SPECIAL_TOKENS, ByteBPETokenizer, fim_inverse,
                                fim_transform, pretokenize)

TEXTS = ["Привіт, світе! Латентна увага.", "Hello, world: MLA & MoE.",
         "def f(x):\n    return x ** 2  # квадрат\n", "Q: 7+8=?\nA: 15", "emoji 🙂 ok",
         ""]


def test_pretokenize_is_lossless():
    for t in TEXTS:
        assert "".join(pretokenize(t)) == t


def test_roundtrip(tok):
    for t in TEXTS:
        assert tok.decode(tok.encode(t)) == t


def test_special_tokens_are_single_ids(tok):
    for s in SPECIAL_TOKENS:
        ids = tok.encode(f"a{s}b")
        assert tok.special_ids[s] in ids
        assert tok.decode(ids) == f"a{s}b"
    assert len(tok.encode("<|eos|>", allow_special=False)) > 1


def test_digits_are_split(tok):
    ids = tok.encode("12345")
    assert len(ids) == 5


def test_training_is_deterministic_and_compresses(tok, tmp_path):
    from conftest import SAMPLE_TEXT
    again = ByteBPETokenizer().train([SAMPLE_TEXT * 3], vocab_size=300)
    assert again.merges == tok.merges
    assert tok.vocab_size == 300
    assert len(tok.encode(SAMPLE_TEXT)) < len(SAMPLE_TEXT.encode("utf-8"))
    tok.save(tmp_path / "t.json")
    loaded = ByteBPETokenizer.load(tmp_path / "t.json")
    assert loaded.encode(SAMPLE_TEXT) == tok.encode(SAMPLE_TEXT)


def test_fertility_positive(tok):
    assert tok.fertility("латентна увага") > 1.0
    assert tok.bytes_per_token("hello world") >= 1.0


def test_fim_transform_roundtrip():
    rng = np.random.default_rng(0)
    text = "def area(r):\n    return 3.14159 * r * r\n"
    seen = 0
    for _ in range(50):
        t = fim_transform(text, 0.5, rng)
        if t.startswith("<|fim_begin|>"):
            seen += 1
            assert "<|fim_hole|>" in t and "<|fim_end|>" in t
        assert fim_inverse(t) == text
    assert 10 < seen < 40
    assert fim_transform(text, 0.0, random.Random(0)) == text
    assert fim_transform(text, 1.0, random.Random(0)).startswith("<|fim_begin|>")
