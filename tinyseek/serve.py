"""A minimal OpenAI-compatible chat server for TinySeek (Chapters 16 and 18).

Built on the standard library (``http.server``) only. Endpoints:

* ``GET  /v1/models``             - the served model;
* ``POST /v1/chat/completions``   - OpenAI chat format: ``messages``,
  ``max_tokens``, ``temperature``, ``top_p``, ``seed``, ``stream``
  (server-sent events). The response includes ``usage`` with token counts,
  so the cost model of Chapter 16 can be applied to a self-hosted model.

The chat template maps every user message to ``Q: ...\\nA: `` (the format
TinySeek saw during SFT). The same client code works against this server
and against the DeepSeek API (``base_url="https://api.deepseek.com"``)::

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
    client.chat.completions.create(model="tinyseek",
        messages=[{"role": "user", "content": "12+34=?"}])

CLI::

    python -m tinyseek.serve --run runs/sft --port 8000
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import torch

from .generate import pick


def chat_template(messages: list[dict]) -> str:
    """Render OpenAI-style messages into a TinySeek prompt."""
    parts = []
    for m in messages:
        role, content = m.get("role"), str(m.get("content", ""))
        if role == "system":
            parts.append(content.strip() + "\n")
        elif role == "user":
            parts.append(f"Q: {content.strip()}\nA: ")
        elif role == "assistant":
            parts.append(content.strip() + "<|eos|>\n")
    return "".join(parts)


class TinySeekService:
    """Model + tokenizer + a lock (CPU inference is serialized) + counters."""

    def __init__(self, model, tok, name: str = "tinyseek") -> None:
        self.model, self.tok, self.name = model.eval(), tok, name
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "busy_seconds": 0.0}

    @torch.no_grad()
    def stream_tokens(self, prompt: str, max_tokens: int, temperature: float,
                      top_p: Optional[float], seed: Optional[int]):
        """Yield generated token ids one by one (cached decoding)."""
        ids = self.tok.encode(prompt)
        ctx = self.model.config.max_position_embeddings
        ids = ids[-(ctx - 1):]
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        cache = self.model.new_cache()
        logits, _ = self.model(torch.tensor([ids]), cache=cache)
        for _ in range(min(max_tokens, ctx - len(ids))):
            nxt = int(pick(logits[:, -1], temperature, None, top_p, gen)[0])
            if nxt == self.tok.eos_id:
                return
            yield nxt
            logits, _ = self.model(torch.tensor([[nxt]]), cache=cache)

    def complete(self, body: dict):
        """Run one chat completion; returns (prompt_ids, generator of tokens)."""
        prompt = chat_template(body.get("messages", []))
        temperature = float(body.get("temperature", 1.0))
        top_p = body.get("top_p")
        max_tokens = int(body.get("max_tokens") or 64)
        return prompt, self.stream_tokens(prompt, max_tokens, temperature,
                                          float(top_p) if top_p is not None else None,
                                          body.get("seed"))


def make_handler(service: TinySeekService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):          # keep the console quiet
            pass

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": service.name, "object": "model", "owned_by": "tinyseek"}]})
            elif self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"status": "ok", **service.stats})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body.get("messages"), list):
                    raise ValueError("'messages' must be a list")
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                return
            rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
            with service.lock:
                t0 = time.time()
                prompt, tokens = service.complete(body)
                n_prompt = len(service.tok.encode(prompt))
                if body.get("stream"):
                    self._stream(rid, created, tokens, n_prompt, t0)
                    return
                out = list(tokens)
                text = service.tok.decode(out)
                finish = "length" if len(out) >= int(body.get("max_tokens") or 64) else "stop"
                self._account(n_prompt, len(out), t0)
            self._json(200, {
                "id": rid, "object": "chat.completion", "created": created,
                "model": service.name,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(out),
                          "total_tokens": n_prompt + len(out)}})

        def _account(self, n_prompt: int, n_out: int, t0: float) -> None:
            s = service.stats
            s["requests"] += 1
            s["prompt_tokens"] += n_prompt
            s["completion_tokens"] += n_out
            s["busy_seconds"] += time.time() - t0

        def _stream(self, rid, created, tokens, n_prompt, t0):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            n_out = 0

            def chunk(delta: dict, finish=None):
                obj = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": service.name,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            chunk({"role": "assistant", "content": ""})
            buf: list[int] = []
            for t in tokens:
                n_out += 1
                buf.append(t)
                text = service.tok.decode(buf)
                if "�" not in text:             # wait for complete UTF-8 characters
                    chunk({"content": text})
                    buf = []
            if buf:
                chunk({"content": service.tok.decode(buf)})
            chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self._account(n_prompt, n_out, t0)
            self.close_connection = True

    return Handler


def make_server(model, tok, host: str = "127.0.0.1", port: int = 8000,
                name: str = "tinyseek") -> ThreadingHTTPServer:
    """Create (but do not start) the HTTP server; ``port=0`` picks a free port.
    The service object is available as ``server.service``."""
    service = TinySeekService(model, tok, name)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.service = service
    return server


def main(argv=None) -> None:
    from .train import load_checkpoint
    p = argparse.ArgumentParser(description="Serve a TinySeek checkpoint with an "
                                "OpenAI-compatible API (Chapters 16, 18).")
    p.add_argument("--run", default="runs/sft")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--name", default="tinyseek")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    model, tok = load_checkpoint(args.run)
    server = make_server(model, tok, args.host, args.port, args.name)
    print(f"serving {args.run} at http://{args.host}:{server.server_address[1]}/v1 "
          "(Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
