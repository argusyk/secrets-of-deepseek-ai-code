"""Fill-in-the-Middle (FIM) transformation in the PSM format.

Teaching code for Chapter 3 of "Secrets of DeepSeek AI". With probability
`rate` a training document is cut at two random points into prefix, middle
and suffix and rewritten as

    <|fim_begin|>prefix<|fim_hole|>suffix<|fim_end|>middle<|eos_token|>

so that the model learns to generate the middle last, after seeing both the
prefix and the suffix. DeepSeek-V3 uses rate = 0.1 (PSM, document level);
DeepSeek-Coder chose 0.5. The real DeepSeek special tokens are written with
full-width bars and the U+2581 block character (see Chapter 17); ASCII
spellings are used here for readability. The same function is part of the
TinySeek package (tinyseek.tokenizer.fim_transform).

Run:
    python fim_psm.py
"""
import random

FIM_BEGIN = "<|fim_begin|>"
FIM_HOLE = "<|fim_hole|>"
FIM_END = "<|fim_end|>"
EOS = "<|eos_token|>"


def fim_transform(text, rate, rng):
    """With probability `rate` rewrite a document as PSM, else keep it."""
    if len(text) < 3 or rng.random() >= rate:
        return text + EOS
    i, j = sorted(rng.sample(range(1, len(text)), 2))
    prefix, middle, suffix = text[:i], text[i:j], text[j:]
    return (FIM_BEGIN + prefix + FIM_HOLE + suffix
            + FIM_END + middle + EOS)


def split_psm(example):
    """Recover (prefix, middle, suffix) from a PSM example."""
    body = example[len(FIM_BEGIN):-len(EOS)]
    prefix, rest = body.split(FIM_HOLE)
    suffix, middle = rest.split(FIM_END)
    return prefix, middle, suffix


if __name__ == "__main__":
    rng = random.Random(0)
    doc = "def mean(xs):\n    total = sum(xs)\n    return total / len(xs)\n"
    ex = fim_transform(doc, rate=1.0, rng=rng)
    print("PSM example:\n" + ex)
    p, m, s = split_psm(ex)
    assert p + m + s == doc, "round trip failed"
    print("\nround trip OK: prefix + middle + suffix == document")

    for rate in (0.1, 0.5):
        rng = random.Random(1)
        n = 10000
        k = sum(fim_transform(doc, rate, rng).startswith(FIM_BEGIN)
                for _ in range(n))
        print(f"rate={rate}: {k / n:.3f} of {n} documents in FIM format")
