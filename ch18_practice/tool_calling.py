"""The agent loop for tool calling, plus an offline mock client.

run_agent() works with any client exposing client.chat.completions.create(...)
in the OpenAI-compatible format: the DeepSeek API, a local server such as
`python -m tinyseek.serve` (both through client.make_client) or MockClient
below. MockClient lets the loop run without network access or an API key.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (offline):
    python tool_calling.py
"""
import json
from types import SimpleNamespace as NS

from review_tools import REGISTRY, TOOLS


def _assistant_to_dict(msg):
    """Convert an API message object into a plain dict for the history.
    Only `content` and `tool_calls` are kept: a reasoning model's
    `reasoning_content` never goes back into the history."""
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
    return d


def run_agent(client, model, messages, tools=TOOLS,
              registry=REGISTRY, max_steps=6, log=print):
    """Call the model, execute requested tools, feed results back, repeat.

    Stops when the model answers without tool calls or after max_steps
    (a hard cap protects against loops and runaway cost).
    """
    messages = list(messages)
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools)
        msg = resp.choices[0].message
        messages.append(_assistant_to_dict(msg))
        if not getattr(msg, "tool_calls", None):
            return msg.content, messages          # the answer is ready
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = registry[tc.function.name](**args)
            except Exception as e:                # the model sees the error
                result = json.dumps({"error": f"{type(e).__name__}: {e}"})
            log(f"[step {step}] {tc.function.name}"
                f"({tc.function.arguments}) -> {len(result)} chars")
            messages.append({"role": "tool",
                             "tool_call_id": tc.id, "content": result})
    return "Stopped: step limit reached.", messages


class MockClient:
    """Deterministic stand-in for the API: first asks for both tools on the
    given file, then writes a review from the tool results in the history."""

    def __init__(self, path="sample_buggy.py"):
        self.chat = NS(completions=NS(create=self._create))
        self.calls = 0
        self.path = path

    def _create(self, model, messages, tools=None, **kw):
        self.calls += 1
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if not tool_msgs:
            args = json.dumps({"path": self.path})
            tcs = [NS(id=f"call_{i}", type="function",
                      function=NS(name=n, arguments=args))
                   for i, n in enumerate(["list_functions", "static_checks"])]
            msg = NS(role="assistant", content=None, tool_calls=tcs)
        else:
            results = [json.loads(m["content"]) for m in tool_msgs[:2]]
            errors = [r["error"] for r in results if isinstance(r, dict)]
            if errors:
                text = "Could not review the file: " + "; ".join(errors)
            else:
                funcs, issues = results
                lines = [f"Reviewed {self.path}: {len(funcs)} functions; "
                         f"found {len(issues)} issues:"]
                lines += [f"- line {i['line']}: {i['kind']} ({i['hint']})"
                          for i in issues]
                no_doc = [f["name"] for f in funcs if not f["has_docstring"]]
                if no_doc:
                    lines.append("- missing docstrings: " + ", ".join(no_doc))
                text = "\n".join(lines)
            msg = NS(role="assistant", content=text, tool_calls=None)
        usage = NS(prompt_tokens=0, completion_tokens=0)
        return NS(choices=[NS(message=msg, finish_reason="stop")],
                  usage=usage)


if __name__ == "__main__":
    client = MockClient()
    answer, history = run_agent(
        client, "mock",
        [{"role": "user",
          "content": "Review sample_buggy.py and list concrete problems."}])
    print(answer)
    print(f"model calls: {client.calls}, history length: {len(history)}")
