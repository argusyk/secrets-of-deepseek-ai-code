"""Structured output with JSON mode, plus validation and one repair attempt.

JSON mode (response_format={"type": "json_object"}) makes the model emit a
syntactically valid JSON object, but it does not enforce *your* schema. The
documentation also asks for the word "json" and an example of the expected
format in the prompt, and warns that the content may occasionally come back
empty. So the program validates the result itself and, if needed, asks once
more with the validation errors attached.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (needs a key, or a local OpenAI-compatible server):
    DEEPSEEK_API_KEY=sk-... python json_mode.py
    python json_mode.py --base-url http://127.0.0.1:8000/v1 --model tinyseek
"""
import argparse
import json
import sys

from client import DEEPSEEK_URL, api_errors, make_client

SYSTEM = """Classify the Python snippet. Reply in json exactly like this:
{"has_bug": true, "category": "mutable-default", "line": 1,
 "explanation": "..."}
category is one of: none, mutable-default, bare-except, eval-exec,
eq-none, other."""

SCHEMA = {"has_bug": bool, "category": str,
          "line": int, "explanation": str}
CATEGORIES = {"none", "mutable-default", "bare-except", "eval-exec",
              "eq-none", "other"}


def validate(obj):
    """Return a list of human-readable problems; empty list means valid."""
    if not isinstance(obj, dict):
        return ["top-level value must be a json object"]
    errs = [f"missing key {k}" for k in SCHEMA if k not in obj]
    errs += [f"{k} must be {t.__name__}" for k, t in SCHEMA.items()
             if k in obj and not isinstance(obj[k], t)]
    if obj.get("category") not in CATEGORIES:
        errs.append("unknown category")
    return errs


def classify(client, snippet, model="deepseek-chat", max_tokens=300):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": snippet}]
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=model, messages=messages,
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
                     {"role": "user", "content": "Fix and reply in json: "
                                                 + "; ".join(errs)}]
    raise ValueError(f"no valid answer after one repair: {errs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=DEEPSEEK_URL)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    client = make_client(a.base_url, timeout=a.timeout)
    snippet = "def f(x, acc=[]):\n    acc.append(x)\n    return acc"
    try:
        print(classify(client, snippet, a.model))
    except (ValueError, *api_errors()) as e:
        sys.exit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
