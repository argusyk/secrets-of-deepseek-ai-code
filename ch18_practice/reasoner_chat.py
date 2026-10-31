"""A multi-turn dialogue with deepseek-reasoner.

The reasoning model returns two fields: `reasoning_content` (the chain of
thought) and `content` (the final answer). Only the final answer goes back
into the history; the API documentation states that reasoning_content must
not be sent back in the input messages. Check the current documentation for
supported parameters: the reasoner has historically accepted fewer sampling
options than deepseek-chat.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (needs a key):
    DEEPSEEK_API_KEY=sk-... python reasoner_chat.py
"""
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")


def ask(history, question):
    messages = history + [{"role": "user", "content": question}]
    resp = client.chat.completions.create(model="deepseek-reasoner", messages=messages)
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    # keep only the final answer in the history
    history = messages + [{"role": "assistant", "content": msg.content}]
    return msg.content, reasoning, history, resp.usage


def show_usage(u):
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    print(f"  prompt={u.prompt_tokens} (cache hit={hit}, miss={miss}) completion={u.completion_tokens}")


if __name__ == "__main__":
    history = []
    q1 = ("A list holds 10 000 sorted integers. How many comparisons does binary "
          "search need in the worst case? Put the final number on the last line.")
    answer, reasoning, history, usage = ask(history, q1)
    print("reasoning (first 300 chars):", (reasoning or "")[:300], "...")
    print("answer:", answer)
    show_usage(usage)

    answer, _, history, usage = ask(history, "And for 1 000 000 integers?")
    print("answer:", answer)
    show_usage(usage)
