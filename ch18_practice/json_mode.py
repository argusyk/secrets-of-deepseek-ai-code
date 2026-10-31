"""Structured output with JSON mode, plus validation and one repair attempt.

JSON mode (response_format={"type": "json_object"}) makes the model emit a
syntactically valid JSON object, but it does not enforce *your* schema. The
documentation also asks for the word "json" and an example of the expected
format in the prompt, and warns that the content may occasionally come back
empty. So the program validates the result itself and, if needed, asks once
more with the validation error attached.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (needs a key):
    DEEPSEEK_API_KEY=sk-... python json_mode.py
"""
import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

SYSTEM = """Classify the Python snippet. Reply in json exactly like this example:
{"has_bug": true, "category": "mutable-default", "line": 1, "explanation": "..."}
category is one of: none, mutable-default, bare-except, eval-exec, eq-none, other."""

SCHEMA = {"has_bug": bool, "category": str, "line": int, "explanation": str}
CATEGORIES = {"none", "mutable-default", "bare-except", "eval-exec", "eq-none", "other"}


def validate(obj):
    """Return a list of human-readable problems; empty list means valid."""
    errs = [f"missing key {k}" for k in SCHEMA if k not in obj]
    errs += [f"{k} must be {t.__name__}" for k, t in SCHEMA.items()
             if k in obj and not isinstance(obj[k], t)]
    if obj.get("category") not in CATEGORIES:
        errs.append("unknown category")
    return errs


def classify(snippet, max_tokens=300):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": snippet}]
    for attempt in range(2):
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages,
            response_format={"type": "json_object"}, max_tokens=max_tokens)
        text = resp.choices[0].message.content or ""
        try:
            obj = json.loads(text)
            errs = validate(obj)
        except json.JSONDecodeError as e:
            obj, errs = None, [f"invalid json: {e}"]
        if not errs:
            return obj
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Fix these problems and reply in json again: "
                                                 + "; ".join(errs)}]
    raise ValueError(f"no valid answer after retry: {errs}")


if __name__ == "__main__":
    print(classify("def f(x, acc=[]):\n    acc.append(x)\n    return acc"))
