"""Building and parsing Fill-in-the-Middle (FIM) prompts in the PSM layout.

DeepSeek-Coder and DeepSeek-V3 were trained with FIM in the
prefix-suffix-middle (PSM) order: the model sees the prefix and the suffix and
generates the missing middle at the end. This file shows how such a prompt is
assembled for a code-completion request inside an editor, and how a training
example is cut from a document. See Chapter 17 (and Chapter 3) of
"Secrets of DeepSeek AI".

The special tokens below are the ones used by DeepSeek-Coder; note the
full-width vertical bars and the U+2581 "lower one eighth block" character.

Run:
    python fim_prompt.py
"""
import random

FIM_BEGIN = "<｜fim▁begin｜>"
FIM_HOLE = "<｜fim▁hole｜>"
FIM_END = "<｜fim▁end｜>"


def psm_prompt(prefix: str, suffix: str) -> str:
    """Prompt for inference: the model must generate the middle after FIM_END."""
    return f"{FIM_BEGIN}{prefix}{FIM_HOLE}{suffix}{FIM_END}"


def split_at_cursor(source: str, cursor: int, selection_end: int | None = None):
    """Editor view: text before the cursor is the prefix, text after the
    (optional) selection is the suffix; the selection itself is to be rewritten."""
    end = cursor if selection_end is None else selection_end
    return source[:cursor], source[end:]


def make_training_example(doc: str, rng: random.Random, fim_rate: float = 0.5):
    """With probability fim_rate turn a document into a PSM example by cutting
    it at two random character positions; otherwise keep it as plain
    next-token data. Returns (text, is_fim)."""
    if len(doc) < 3 or rng.random() >= fim_rate:
        return doc, False
    i, j = sorted(rng.sample(range(1, len(doc)), 2))
    prefix, middle, suffix = doc[:i], doc[i:j], doc[j:]
    return psm_prompt(prefix, suffix) + middle, True


def extract_middle(prompt: str, completion: str) -> str:
    """Insert the generated middle back between prefix and suffix."""
    body = prompt[len(FIM_BEGIN):-len(FIM_END)]
    prefix, suffix = body.split(FIM_HOLE, 1)
    return prefix + completion + suffix


if __name__ == "__main__":
    source = (
        "def mean(xs):\n"
        "    if not xs:\n"
        "        raise ValueError('empty')\n"
        "    \n"
        "    return total / len(xs)\n"
    )
    cursor = source.index("    \n") + 4          # the empty line in the body
    prefix, suffix = split_at_cursor(source, cursor)
    prompt = psm_prompt(prefix, suffix)
    print("PSM prompt (repr):")
    print(repr(prompt))

    completion = "total = sum(xs)"           # what a model would produce
    print("\nreassembled function:")
    print(extract_middle(prompt, completion))

    rng = random.Random(0)
    docs = ["x = 1\ny = x + 1\nprint(y)\n"] * 1000
    n_fim = sum(make_training_example(d, rng)[1] for d in docs)
    print(f"FIM share in a toy corpus at fim_rate=0.5: {n_fim / len(docs):.3f}")
