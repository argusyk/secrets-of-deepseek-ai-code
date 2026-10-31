"""Rule-based rewards in the style of DeepSeek-R1-Zero.

Teaching implementation for Chapter 13 of "Secrets of DeepSeek AI".
Two components are combined:
  * format reward  - the completion must follow <think>...</think><answer>...</answer>;
  * accuracy reward - the extracted answer must match the reference.
A third, optional component reproduces the idea of the language-consistency
reward used for DeepSeek-R1: the share of reasoning words written in the
target script.

Run:
    python rule_reward.py
"""
import re

TEMPLATE = re.compile(r"^\s*<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$", re.DOTALL)


def format_reward(completion: str) -> float:
    return 1.0 if TEMPLATE.match(completion) else 0.0


def normalize(ans: str) -> str:
    ans = ans.strip().strip(".").replace(" ", "").replace(",", ".")
    try:
        return repr(float(ans))                 # "12", "12.0", "12.00" -> same key
    except ValueError:
        return ans.lower()


def accuracy_reward(completion: str, reference: str) -> float:
    m = TEMPLATE.match(completion)
    if not m:
        return 0.0                              # unparseable answer earns nothing
    return 1.0 if normalize(m.group(2)) == normalize(reference) else 0.0


def language_consistency(completion: str, script: str = "latin") -> float:
    """Share of words in the reasoning written in the target script."""
    m = TEMPLATE.match(completion)
    text = m.group(1) if m else completion
    words = re.findall(r"[^\W\d_]+", text)
    if not words:
        return 0.0
    pat = r"[A-Za-z]+" if script == "latin" else r"[Ѐ-ӿ]+"
    hits = sum(1 for w in words if re.fullmatch(pat, w))
    return hits / len(words)


def reward(completion: str, reference: str, use_language=False) -> float:
    r = accuracy_reward(completion, reference) + format_reward(completion)
    if use_language:
        r += language_consistency(completion)
    return r


if __name__ == "__main__":
    ref = "12"
    cases = {
        "correct, well formed": "<think>3 boxes of 4 apples: 3*4 = 12</think><answer>12</answer>",
        "correct, 12.0 form":   "<think>3 times 4</think><answer>12.0</answer>",
        "wrong answer":         "<think>3 plus 4 is 7</think><answer>7</answer>",
        "no tags":              "The answer is 12.",
        "mixed languages":      "<think>3 коробки по 4 apples, тобто 12</think><answer>12</answer>",
    }
    for name, c in cases.items():
        print(f"{name:22s} format={format_reward(c):.0f} acc={accuracy_reward(c, ref):.0f} "
              f"lang={language_consistency(c):.2f} total={reward(c, ref, True):.2f}")
