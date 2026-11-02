"""One place to create an OpenAI-compatible client (Chapter 16).

Every API example of the book builds its client here, so that all of them
share the same safe defaults:

* an explicit timeout and a small number of SDK retries (the SDK defaults
  are a 600-second timeout and 2 retries; a teaching example that preaches
  "a timeout on every request" should set its own);
* a clear message instead of a traceback when the key or the `openai`
  package is missing;
* one switch between the DeepSeek API and any other OpenAI-compatible
  server, for instance a local `python -m tinyseek.serve` (no key needed).

    from client import make_client
    client = make_client()                                   # DeepSeek API
    local = make_client("http://127.0.0.1:8000/v1")          # TinySeek

Run this file to check the environment without sending any request:
    python client.py
"""
import os
import sys

DEEPSEEK_URL = "https://api.deepseek.com"
KEY_ENV = "DEEPSEEK_API_KEY"


def make_client(base_url=DEEPSEEK_URL, timeout=60.0, max_retries=3,
                key_env=KEY_ENV):
    """OpenAI client with a timeout, bounded retries and a clear key check.

    For the DeepSeek API the key is taken from the environment variable
    `key_env`; a local server (any other base_url) accepts a dummy key.
    Exits with a one-line message instead of a traceback on a missing key.
    """
    key = os.environ.get(key_env, "").strip()
    if not key:
        if base_url.rstrip("/") == DEEPSEEK_URL:
            sys.exit(f"{key_env} is not set. Get a key at "
                     "platform.deepseek.com and export it, or pass "
                     "--base-url for a local OpenAI-compatible server.")
        key = "none"
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The 'openai' package is missing: "
                 "pip install 'openai>=1.0'")
    return OpenAI(api_key=key, base_url=base_url, timeout=timeout,
                  max_retries=max_retries)


def describe(base_url=DEEPSEEK_URL, key_env=KEY_ENV):
    """A short status line about the environment (no network access)."""
    has_key = bool(os.environ.get(key_env, "").strip())
    try:
        import openai
        sdk = f"openai {openai.__version__}"
    except ImportError:
        sdk = "openai not installed"
    return f"base_url={base_url}  {key_env} set: {has_key}  {sdk}"


if __name__ == "__main__":
    print(describe())
    ready = bool(os.environ.get(KEY_ENV, "").strip())
    print("make_client() would",
          "connect" if ready else "stop with a message asking for the key")
