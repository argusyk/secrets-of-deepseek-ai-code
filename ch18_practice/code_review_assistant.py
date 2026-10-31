"""The code-review assistant: the through-line mini-project of Chapter 18.

Uses the real DeepSeek API when DEEPSEEK_API_KEY is set and --mock is not
given; otherwise falls back to the offline MockClient, so the whole pipeline
can be run without network access.

Run:
    python code_review_assistant.py sample_buggy.py --mock
    DEEPSEEK_API_KEY=sk-... python code_review_assistant.py sample_buggy.py
"""
import argparse
import os

from tool_calling import MockClient, run_agent

SYSTEM = (
    "You are a careful Python code reviewer. Use the tools to inspect the file "
    "before answering. Report only problems you can point to by line number, "
    "then suggest a fix for each. Be concise."
)


def make_client(use_mock):
    if use_mock or not os.environ.get("DEEPSEEK_API_KEY"):
        return MockClient(), "mock"
    from openai import OpenAI          # imported lazily: not needed offline
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                  base_url="https://api.deepseek.com"), "deepseek-chat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    client, model = make_client(a.mock)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Review the file {a.path}."}]
    answer, _ = run_agent(client, model, messages, max_steps=6)
    print(f"--- review ({model}) ---")
    print(answer)


if __name__ == "__main__":
    main()
