"""The code-review assistant: the running example of Chapter 18.

One script, three backends, the same client code:

    python code_review_assistant.py sample_buggy.py --mock     # offline
    DEEPSEEK_API_KEY=sk-... python code_review_assistant.py sample_buggy.py
    python code_review_assistant.py sample_buggy.py \\
        --base-url http://127.0.0.1:8000/v1 --model tinyseek   # TinySeek

The file must lie inside the tools' sandbox (this directory, see
review_tools.SANDBOX); the model only ever names it relative to the sandbox.
A backend may ignore the `tools` parameter (the tiny `tinyseek.serve` model
does): the loop then ends after one step, and the assistant appends the
facts from the tools itself, so the report stays checkable.
"""
import argparse
import sys

from client import DEEPSEEK_URL, api_errors, make_client
from review_tools import REGISTRY, sandbox_relpath
from tool_calling import MockClient, run_agent

SYSTEM = (
    "You are a careful Python code reviewer. Use the tools to inspect the "
    "file before answering. Report only problems you can point to by line "
    "number, then suggest a fix for each. Be concise."
)


def tool_facts(rel):
    """The deterministic part of a review, without any model."""
    return (f"Facts from the tools for {rel}:\n"
            f"list_functions: {REGISTRY['list_functions'](rel)}\n"
            f"static_checks: {REGISTRY['static_checks'](rel)}")


def review(client, model, rel, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Review the file {rel}."}]
    answer, history = run_agent(client, model, messages,
                                max_steps=max_steps)
    if not any(m["role"] == "tool" for m in history):
        answer = f"{answer}\n\n[the model called no tools]\n{tool_facts(rel)}"
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="Python file inside the sandbox")
    ap.add_argument("--mock", action="store_true", help="offline MockClient")
    ap.add_argument("--base-url", default=DEEPSEEK_URL)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-steps", type=int, default=6)
    a = ap.parse_args()
    try:
        rel = sandbox_relpath(a.path)
    except ValueError as e:
        ap.error(str(e))
    if a.mock:
        client, model = MockClient(rel), "mock"
    else:
        client, model = make_client(a.base_url, timeout=a.timeout), a.model
    try:
        answer = review(client, model, rel, a.max_steps)
    except api_errors() as e:
        sys.exit(f"request failed: {type(e).__name__}: {e}")
    print(f"--- review ({model}) ---")
    print(answer)


if __name__ == "__main__":
    main()
