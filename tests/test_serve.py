"""The OpenAI-compatible server (Chapters 16 and 18)."""
import json
import threading
import urllib.request

from conftest import small_config
from tinyseek.model import TinySeekModel
from tinyseek.serve import chat_template, make_server


def test_chat_template():
    p = chat_template([{"role": "user", "content": "2+2=?"}])
    assert p == "Q: 2+2=?\nA: "


def test_chat_completion_roundtrip(tok):
    server = make_server(TinySeekModel(small_config()), tok, port=0)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        with urllib.request.urlopen(url + "/models") as r:
            assert json.load(r)["data"][0]["id"] == "tinyseek"
        body = json.dumps({"model": "tinyseek", "max_tokens": 5, "temperature": 0,
                           "messages": [{"role": "user", "content": "2+2=?"}]}).encode()
        req = urllib.request.Request(url + "/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            out = json.load(r)
        assert out["object"] == "chat.completion"
        assert out["choices"][0]["message"]["role"] == "assistant"
        assert out["usage"]["completion_tokens"] <= 5
        body = json.dumps({"stream": True, "max_tokens": 3, "temperature": 0,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(url + "/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            text = r.read().decode()
        assert text.strip().endswith("data: [DONE]")
    finally:
        server.shutdown()
        server.server_close()
