"""Code completion with Fill-in-the-Middle: the DeepSeek API or TinySeek.

The DeepSeek platform exposes FIM completion on its beta endpoint with an
OpenAI-compatible `completions` call: you pass the text before the cursor as
`prompt` and the text after it as `suffix`. The API assembles the FIM prompt
itself, so you do not write special tokens by hand. The client gets a timeout
and a few SDK retries with exponential backoff; a missing key ends with a
one-line message, not a traceback.

Without a key, the same prefix and suffix can be completed by a local TinySeek
checkpoint (Section 17.9): the PSM prompt is then built here with TinySeek's
own FIM tokens. The miniature model restores short identifiers at best.

See Chapter 17 of "Secrets of DeepSeek AI" and https://api-docs.deepseek.com/.

Run:
    export DEEPSEEK_API_KEY="sk-..."
    python code_completion_api.py
    python code_completion_api.py --tinyseek ../runs/v3      # no key needed
"""
import argparse
import os
import sys
from pathlib import Path

BETA_URL = "https://api.deepseek.com/beta"


def make_client(timeout=30.0, max_retries=3):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set: export DEEPSEEK_API_KEY=sk-... "
                 "or run with --tinyseek RUN_DIR for a local model")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The 'openai' package is missing: pip install 'openai>=1.0'")
    # the SDK retries 429, 5xx and timeouts with exponential backoff
    return OpenAI(api_key=key, base_url=BETA_URL, timeout=timeout,
                  max_retries=max_retries)


def complete_middle(client, prefix, suffix, max_tokens=128,
                    model="deepseek-chat"):
    resp = client.completions.create(
        model=model, prompt=prefix, suffix=suffix,
        max_tokens=max_tokens, temperature=0.0)
    return resp.choices[0].text


def complete_middle_tinyseek(run_dir, prefix, suffix, max_tokens=16):
    """Greedy FIM completion with a TinySeek checkpoint (PSM order)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tinyseek.evaluate import complete_prompts
    from tinyseek.train import load_checkpoint
    model, tok = load_checkpoint(run_dir)
    prompt = f"<|fim_begin|>{prefix}<|fim_hole|>{suffix}<|fim_end|>"
    return complete_prompts(model, tok, [prompt], max_tokens)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tinyseek", metavar="RUN_DIR",
                    help="complete with a local TinySeek checkpoint")
    a = ap.parse_args()
    prefix = 'def fib(n):\n    """Return the n-th Fibonacci number."""\n'
    suffix = "\n\nprint([fib(i) for i in range(10)])\n"
    if a.tinyseek:
        middle = complete_middle_tinyseek(a.tinyseek, prefix, suffix)
    else:
        client = make_client()
        import openai
        try:
            middle = complete_middle(client, prefix, suffix)
        except openai.APIError as e:    # after the SDK's own retries
            sys.exit(f"request failed: {type(e).__name__}: {e}")
    print(prefix + middle + suffix)


if __name__ == "__main__":
    main()
