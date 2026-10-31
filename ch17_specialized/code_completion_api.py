"""Code completion with Fill-in-the-Middle through the DeepSeek API (beta).

The DeepSeek platform exposes FIM completion on its beta endpoint with an
OpenAI-compatible `completions` call: you pass the text before the cursor as
`prompt` and the text after it as `suffix`. The API assembles the FIM prompt
itself, so you do not write special tokens by hand. Requires DEEPSEEK_API_KEY.
See Chapter 17 of "Secrets of DeepSeek AI" and https://api-docs.deepseek.com/.

Run:
    export DEEPSEEK_API_KEY="sk-..."
    python code_completion_api.py
"""
import os

from openai import OpenAI


def complete_middle(prefix: str, suffix: str, max_tokens: int = 128) -> str:
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com/beta")
    resp = client.completions.create(
        model="deepseek-chat",
        prompt=prefix,
        suffix=suffix,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].text


if __name__ == "__main__":
    prefix = "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
    suffix = "\n\nprint([fib(i) for i in range(10)])\n"
    middle = complete_middle(prefix, suffix)
    print(prefix + middle + suffix)
