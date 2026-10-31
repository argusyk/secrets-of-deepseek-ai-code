"""The agent loop for tool calling, plus an offline mock client.

run_agent() works with any client exposing client.chat.completions.create(...)
in the OpenAI-compatible format used by the DeepSeek API: the real
`openai.OpenAI(base_url="https://api.deepseek.com")` or MockClient below.
MockClient lets the loop run without network access or an API key.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (offline):
    python tool_calling.py
"""
import json
from types import SimpleNamespace as NS

from review_tools import REGISTRY, TOOLS


def _assistant_to_dict(msg):
    """Convert an API message object into a plain dict for the history."""
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
    return d


def run_agent(client, model, messages, tools=TOOLS, registry=REGISTRY,
              max_steps=6, log=print):
    """Call the model, execute requested tools, feed results back, repeat.

    Stops when the model answers without tool calls or after max_steps
    (a hard cap protects against loops and runaway cost).
    """
    messages = list(messages)
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools)
        msg = resp.choices[0].message
        messages.append(_assistant_to_dict(msg))
        if not getattr(msg, "tool_calls", None):
            return msg.content, messages
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = registry[name]                 # unknown names raise KeyError
                result = fn(**args)
            except Exception as e:                  # the error goes back to the model
                result = json.dumps({"error": f"{type(e).__name__}: {e}"})
            log(f"[step {step}] tool {name}({tc.function.arguments}) -> {len(result)} chars")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Stopped: step limit reached.", messages


class MockClient:
    """Deterministic stand-in for the API: first asks for both tools, then
    writes a review from the tool results it finds in the history."""

    def __init__(self):
        self.chat = NS(completions=NS(create=self._create))
        self.calls = 0

    def _create(self, model, messages, tools=None, **kw):
        self.calls += 1
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if not tool_msgs:
            path = "sample_buggy.py"
            tcs = [NS(id=f"call_{i}", type="function",
                      function=NS(name=n, arguments=json.dumps({"path": path})))
                   for i, n in enumerate(["list_functions", "static_checks"])]
            msg = NS(role="assistant", content=None, tool_calls=tcs)
        else:
            funcs = json.loads(tool_msgs[0]["content"])
            issues = json.loads(tool_msgs[1]["content"])
            lines = [f"Reviewed {len(funcs)} functions; found {len(issues)} issues:"]
            lines += [f"- line {i['line']}: {i['kind']} ({i['hint']})" for i in issues]
            no_doc = [f["name"] for f in funcs if not f["has_docstring"]]
            if no_doc:
                lines.append("- missing docstrings: " + ", ".join(no_doc))
            msg = NS(role="assistant", content="\n".join(lines), tool_calls=None)
        usage = NS(prompt_tokens=0, completion_tokens=0)
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=usage)


if __name__ == "__main__":
    client = MockClient()
    answer, history = run_agent(
        client, "deepseek-chat",
        [{"role": "user", "content": "Review sample_buggy.py and list concrete problems."}])
    print(answer)
    print(f"model calls: {client.calls}, history length: {len(history)}")
