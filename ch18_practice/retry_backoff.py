"""Retries with exponential backoff and full jitter.

Retry only what can succeed on a second try: rate limiting (429), server
errors (500, 502, 503, 504), timeouts and broken connections. Do not retry
400 (bad request), 401 (bad key), 402 (no balance) or 422 (bad parameters):
repeating them only burns time. The delay before retry k is uniform in
[0, min(cap, base * 2**k)] (formula (18.1)); a Retry-After header sent by the
server takes precedence.

Works with the exceptions of the `openai` SDK (they carry `status_code` and
the HTTP `response`) and with the small stand-ins below, so the demo runs
offline. client.py wraps every SDK request with call_with_retries and sets
the SDK's own max_retries=0, so the retries do not multiply.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (offline):
    python retry_backoff.py
"""
import random
import time
from types import SimpleNamespace as NS

RETRYABLE = {429, 500, 502, 503, 504}

try:                                   # timeouts and broken connections
    import openai
    NETWORK_ERRORS = (TimeoutError, ConnectionError,
                      openai.APIConnectionError)   # incl. APITimeoutError
except ImportError:
    NETWORK_ERRORS = (TimeoutError, ConnectionError)


class APIError(Exception):
    """Offline stand-in for an HTTP error of the SDK."""

    def __init__(self, status_code, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = NS(headers=headers or {})


def backoff_delay(attempt, base=0.5, cap=20.0, rng=random):
    """Full jitter: uniform in [0, min(cap, base * 2**attempt)]."""
    return rng.uniform(0, min(cap, base * 2 ** attempt))


def retry_after(exc):
    """Seconds from a Retry-After header, if the server sent one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def call_with_retries(fn, max_attempts=5, base=0.5, cap=20.0,
                      sleep=time.sleep, rng=random, log=print):
    """Call fn(); retry transient failures, re-raise permanent ones."""
    for attempt in range(max_attempts):
        wait = None
        try:
            return fn()
        except NETWORK_ERRORS as e:            # timeout, connection reset
            status, last = type(e).__name__, e
        except Exception as e:
            status, last = getattr(e, "status_code", None), e
            if status not in RETRYABLE:
                raise                          # permanent error: fail fast
            wait = retry_after(e)              # the server named the pause
        if attempt == max_attempts - 1:
            raise RuntimeError(f"gave up after {max_attempts} attempts "
                               f"(last: {status})") from last
        d = wait if wait is not None else backoff_delay(attempt, base,
                                                        cap, rng)
        log(f"attempt {attempt + 1} failed ({status}); "
            f"retrying in {d:.2f}s")
        sleep(d)


def always_401():
    raise APIError(401)


if __name__ == "__main__":
    rng = random.Random(42)
    script = iter([APIError(503), TimeoutError(),
                   APIError(429, {"retry-after": "2"}), "OK: review ready"])

    def flaky():
        x = next(script)
        if isinstance(x, BaseException):
            raise x
        return x

    slept = []
    print(call_with_retries(flaky, sleep=slept.append, rng=rng))
    print(f"total simulated wait: {sum(slept):.2f}s")

    try:
        call_with_retries(always_401, sleep=slept.append, rng=rng)
    except APIError as e:
        print("not retried:", e)
