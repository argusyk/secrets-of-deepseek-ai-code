"""One OpenAI-compatible client for every API example of Chapter 18.

The same code talks to three backends:

* the DeepSeek API (default; needs DEEPSEEK_API_KEY);
* any other OpenAI-compatible server, for instance a local TinySeek model
  served with `python -m tinyseek.serve` (no key needed):
      make_client("http://127.0.0.1:8000/v1")
* the offline MockClient of tool_calling.py (the --mock flag of the scripts).

Safe defaults, as the checklist of Section 18.7 asks:

* an explicit timeout on every request (the SDK default is 600 seconds);
* retries with exponential backoff and full jitter done by our own
  `call_with_retries` (retry_backoff.py); the SDK's built-in retries are
  switched off (max_retries=0) so that the two do not multiply;
* a one-line message instead of a traceback when the key or the `openai`
  package is missing.

Chapter 16 has the same factory in ch16_deployment/client.py; this copy keeps
the chapter directory self-contained.

Run this file to check the environment without sending any request:
    python client.py
"""
import os
import sys
from types import SimpleNamespace as NS

from retry_backoff import call_with_retries

DEEPSEEK_URL = "https://api.deepseek.com"
TINYSEEK_URL = "http://127.0.0.1:8000/v1"
KEY_ENV = "DEEPSEEK_API_KEY"


class RetryingClient:
    """Wraps an OpenAI client: chat.completions.create goes through
    call_with_retries. The raw client stays available as `.raw`."""

    def __init__(self, raw, **retry_kw):
        self.raw = raw
        create = raw.chat.completions.create

        def create_with_retries(**kw):
            return call_with_retries(lambda: create(**kw), **retry_kw)

        self.chat = NS(completions=NS(create=create_with_retries))


def make_client(base_url=DEEPSEEK_URL, timeout=60.0, key_env=KEY_ENV,
                **retry_kw):
    """OpenAI-compatible client with a timeout and our own retries.

    The DeepSeek API needs the key from `key_env`; a local server (any other
    base_url) accepts a dummy key. Exits with a one-line message instead of
    a traceback when the key or the openai package is missing.
    """
    key = os.environ.get(key_env, "").strip()
    if not key:
        if base_url.rstrip("/") == DEEPSEEK_URL:
            sys.exit(f"{key_env} is not set. Get a key at "
                     f"platform.deepseek.com, then: export {key_env}=sk-... "
                     "(or use --mock, or --base-url for a local server)")
        key = "none"
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The 'openai' package is missing: pip install 'openai>=1.0'")
    raw = OpenAI(api_key=key, base_url=base_url, timeout=timeout,
                 max_retries=0)
    retry_kw.setdefault("log", lambda s: print(s, file=sys.stderr))
    return RetryingClient(raw, **retry_kw)


def api_errors():
    """Exception types a script may turn into a short message on exit."""
    try:
        import openai
        return (RuntimeError, openai.APIError)
    except ImportError:
        return (RuntimeError,)


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
    print(f"a local TinySeek server needs no key: make_client({TINYSEEK_URL!r})")
