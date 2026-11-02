"""Guardrail around a TinySeek server: the TinySeek step of Chapter 19.

Starts the OpenAI-compatible server of `tinyseek.serve` in-process on a
free port and talks to it only through `GuardedClient` (Listing 19.3),
which applies the `Guardrail` of guardrail.py on both sides of the call:

  * the user text and every retrieved document are screened and redacted;
  * documents are passed to the model as marked, untrusted data;
  * the system prompt carries a canary; the answer is checked for it and
    redacted before it reaches the user;
  * TinySeek has no tool API, so the tool layer of injection_tests.py
    (allow-list plus confirmation) is not exercised here.

The script reports three measurements:

  1. the attack suite of injection_tests.py replayed against the real
     server: which layer acted and what actually reached the model;
  2. false positives on benign traffic: TinySeek's own arithmetic prompts
     plus business text with dates, order and invoice numbers;
  3. the latency overhead of the guardrail relative to the model call.

A tiny model does not follow injected instructions anyway, which is the
point: the measurement is taken at the boundary of the system, not from
the model's good behaviour. The checkpoint of lab 6 or lab 4 is used if
it exists (code/runs/...); otherwise a randomly initialised tiny model
is served, which is enough to test the plumbing.

Run:
    python guarded_tinyseek.py
"""
import dataclasses
import json
import logging
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from guardrail import Guardrail, Verdict           # noqa: E402
from injection_tests import SYSTEM, TESTS              # noqa: E402

CHECKPOINTS = ("runs/lab6/sft", "runs/base_sft", "runs/sft",
               "runs/lab4/v3", "runs/base_v3", "runs/v3")


def load_model():
    """A trained TinySeek checkpoint if there is one, else a random one."""
    from tinyseek.train import load_checkpoint
    for d in CHECKPOINTS:
        if (CODE / d / "model.pt").exists():
            model, tok = load_checkpoint(CODE / d)
            if tok is not None:
                return model, tok, d
    import torch
    from tinyseek.config import get_preset
    from tinyseek.model import TinySeekModel
    from tinyseek.tokenizer import ByteBPETokenizer
    torch.manual_seed(0)
    text = ("Q: 47+38=?\nA: <think>7+8=15 4+3+1=8</think>"
            "<answer>85</answer>\n") * 20
    tok = ByteBPETokenizer().train([text], vocab_size=300)
    model = TinySeekModel(get_preset("tiny_v3",
                                     vocab_size=tok.vocab_size))
    return model, tok, "random tiny_v3 (no checkpoint found)"


# >>> listing 19.3
class GuardedClient:
    """OpenAI-style chat client with a guardrail on both sides."""

    def __init__(self, base_url, guard, timeout=30.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.guard, self.timeout = guard, timeout
        self.model_seconds = 0.0

    def _post(self, messages, max_tokens):
        body = json.dumps({"model": "tinyseek", "messages": messages,
                           "max_tokens": max_tokens,
                           "temperature": 0}).encode()
        req = urllib.request.Request(
            self.url, body, {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        self.model_seconds += time.perf_counter() - t0
        return out["choices"][0]["message"]["content"]

    def chat(self, user, documents=(), max_tokens=24):
        v = self.guard.check_input(user)
        if not v.allowed:
            return Verdict(False, "Request blocked by policy.", v.reasons)
        reasons, docs = list(v.reasons), []
        for d in documents:                    # untrusted content
            dv = self.guard.check_input(d)
            if dv.allowed:
                docs.append(dv.text)
            else:
                reasons.append("document dropped")
        data = "".join(f"\n<document>{d}</document>" for d in docs)
        messages = [
            {"role": "system",
             "content": self.guard.system_prompt(SYSTEM)},
            {"role": "user", "content": v.text + data}]
        out = self.guard.check_output(self._post(messages, max_tokens))
        return Verdict(out.allowed, out.text, reasons + out.reasons)
# <<< listing 19.3


BENIGN_BUSINESS = [
    "Order 1234567890 shipped on 2024-01-15, invoice 2024/0457.",
    "Meeting at 14:30 on 15.01.2025, room 101.",
    "Ticket 42: the build 3.12.7 fails on line 1048.",
    "Version 20250115-1 of the price list, 12 345 items.",
    "Call the reception on extension 4567.",
]


def main():
    logging.basicConfig(level=logging.WARNING)
    from tinyseek.serve import make_server
    model, tok, source = load_model()
    server = make_server(model, tok, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    print(f"model: {source}\nserver: {base}\n")
    g = Guardrail()
    client = GuardedClient(base, g)
    try:
        # 1. attack suite against the real server
        print(f"{'test':<24}{'reached model':<15}layers that acted")
        for name, user, docs in TESTS:
            v = client.chat(user, docs)
            reached = "no" if v.text == "Request blocked by policy." \
                else "yes"
            acted = ", ".join(v.reasons) or "-"
            print(f"{name:<24}{reached:<15}{acted}")

        # 2. false positives on benign traffic
        rng = random.Random(0)
        prompts = [f"{rng.randrange(10**4)}+{rng.randrange(10**4)}=?"
                   for _ in range(200)] + BENIGN_BUSINESS
        flagged = [p for p in prompts
                   if g.check_input(p).reasons]
        print(f"\nbenign prompts: {len(prompts)}, "
              f"false positives: {len(flagged)}")
        for p in flagged:
            print("  flagged:", p)

        # 3. latency of the guardrail vs the model call
        n = 20
        client.model_seconds = 0.0
        t0 = time.perf_counter()
        for p in prompts[:n]:
            client.chat(p, max_tokens=16)
        total = time.perf_counter() - t0
        guard_ms = (total - client.model_seconds) / n * 1000
        model_ms = client.model_seconds / n * 1000
        print(f"\nper request: model {model_ms:.1f} ms, guardrail and "
              f"client {guard_ms:.2f} ms "
              f"({guard_ms / max(model_ms, 1e-9):.1%} of model time)")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
