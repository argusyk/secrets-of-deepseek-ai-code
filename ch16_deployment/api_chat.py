"""Minimal DeepSeek API calls: a plain request, streaming, and a reasoner call.

Chapter 16 of "Secrets of DeepSeek AI". The DeepSeek HTTP API is
OpenAI-compatible, so the official `openai` client works with a different
base_url. Set the key first:

    export DEEPSEEK_API_KEY="sk-..."
    python api_chat.py

Model names (`deepseek-chat`, `deepseek-reasoner`) and response fields follow
the provider's documentation at the time of writing; check it for changes.
"""
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com")

SYSTEM = "You are a concise assistant for software engineers."


def ask(question: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0.3,
    )
    u = resp.usage
    # cache-hit/miss counters are provider-specific extensions of `usage`
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    print(f"[usage] prompt={u.prompt_tokens} completion={u.completion_tokens} cache_hit={hit}")
    return resp.choices[0].message.content


def ask_stream(question: str) -> None:
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": question}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def ask_reasoner(question: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": question}],
    )
    msg = resp.choices[0].message
    thinking = getattr(msg, "reasoning_content", "") or ""
    print(f"[reasoning: {len(thinking)} chars, billed as output tokens]")
    return msg.content


if __name__ == "__main__":
    print(ask("What does MLA cache instead of full keys and values? One sentence."))
    ask_stream("List three costs that an API price per token hides.")
    print(ask_reasoner("Is 2**61 - 1 prime? Answer yes or no."))
