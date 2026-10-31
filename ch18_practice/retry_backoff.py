"""Retries with exponential backoff and full jitter.

Retry only what can succeed on a second try: rate limiting (429), server
errors (500, 502, 503, 504) and timeouts. Do not retry 400 (bad request),
401 (bad key), 402 (no balance) or 422 (bad parameters): repeating them only
burns time. The delay before attempt k is uniform in [0, min(cap, base*2^k)].

See Chapter 18 of "Secrets of DeepSeek AI".

Run (offline):
    python retry_backoff.py
"""
import random
import time

RETRYABLE = {429, 500, 502, 503, 504}


class APIError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


def backoff_delay(attempt, base=0.5, cap=20.0, rng=random):
    """Full jitter: uniform in [0, min(cap, base * 2**attempt)]."""
    return rng.uniform(0, min(cap, base * 2 ** attempt))


def call_with_retries(fn, max_attempts=5, base=0.5, cap=20.0,
                      sleep=time.sleep, rng=random, log=print):
    for attempt in range(max_attempts):
        try:
            return fn()
        except TimeoutError:
            status = "timeout"
        except APIError as e:
            if e.status not in RETRYABLE:
                raise                        # permanent error: fail fast
            status = e.status
        if attempt == max_attempts - 1:
            raise RuntimeError(f"gave up after {max_attempts} attempts (last: {status})")
        d = backoff_delay(attempt, base, cap, rng)
        log(f"attempt {attempt + 1} failed ({status}); retrying in {d:.2f}s")
        sleep(d)


if __name__ == "__main__":
    rng = random.Random(42)
    script = iter([APIError(503), TimeoutError(), APIError(429), "OK: review ready"])

    def flaky():
        x = next(script)
        if isinstance(x, BaseException):
            raise x
        return x

    slept = []
    print(call_with_retries(flaky, sleep=slept.append, rng=rng))
    print(f"total simulated wait: {sum(slept):.2f}s")

    try:
        call_with_retries(lambda: (_ for _ in ()).throw(APIError(401)), sleep=slept.append, rng=rng)
    except APIError as e:
        print("not retried:", e)
