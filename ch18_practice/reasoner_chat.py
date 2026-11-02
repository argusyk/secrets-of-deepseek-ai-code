"""A multi-turn dialogue with deepseek-reasoner.

The reasoning model returns two fields: `reasoning_content` (the chain of
thought) and `content` (the final answer). Only the final answer goes back
into the history; the API documentation states that reasoning_content must
not be sent back in the input messages of the next turn. Check the current
documentation for supported parameters: the reasoner has historically
accepted fewer sampling options than deepseek-chat.

See Chapter 18 of "Secrets of DeepSeek AI".

Run (needs a key):
    DEEPSEEK_API_KEY=sk-... python reasoner_chat.py
"""
import sys

from client import api_errors, make_client


def ask(client, history, question, model="deepseek-reasoner"):
    messages = history + [{"role": "user", "content": question}]
    resp = client.chat.completions.create(model=model, messages=messages)
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    # only the final answer goes into the history
    history = messages + [{"role": "assistant", "content": msg.content}]
    return msg.content, reasoning, history, resp.usage


def show_usage(u):
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    print(f"  prompt={u.prompt_tokens} (cache hit={hit}, miss={miss}) "
          f"completion={u.completion_tokens}")


def main():
    client = make_client(timeout=300.0)       # reasoning can take minutes
    q1 = ("A list holds 10 000 sorted integers. How many comparisons does "
          "binary search need in the worst case? Put the final number on "
          "the last line.")
    try:
        answer, reasoning, history, usage = ask(client, [], q1)
        print("reasoning (first 300 chars):", (reasoning or "")[:300], "...")
        print("answer:", answer)
        show_usage(usage)
        answer, _, history, usage = ask(client, history,
                                        "And for 1 000 000 integers?")
        print("answer:", answer)
        show_usage(usage)
    except api_errors() as e:
        sys.exit(f"request failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
