"""Minimal DeepSeek API calls: a plain request, streaming, and a reasoner.

Chapter 16 of "Secrets of DeepSeek AI". The DeepSeek HTTP API is
OpenAI-compatible, so the official `openai` client works with a different
base_url. The client comes from `client.make_client` (timeout, bounded
retries, a clear message when the key is missing). Set the key first:

    export DEEPSEEK_API_KEY="sk-..."
    python api_chat.py

The same script talks to a local TinySeek server (Section 16.8), no key:

    python -m tinyseek.serve --run runs/sft --port 8000     # from code/
    python api_chat.py --base-url http://127.0.0.1:8000/v1 --model tinyseek

Model names (`deepseek-chat`, `deepseek-reasoner`) and response fields follow
the provider's documentation at the time of writing; check it for changes.
"""
import argparse

from client import DEEPSEEK_URL, make_client

SYSTEM = "You are a concise assistant for software engineers."


def ask(client, question, model="deepseek-chat"):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0.3,
    )
    u = resp.usage
    # cache-hit/miss counters are provider-specific extensions of `usage`
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    print(f"[usage] prompt={u.prompt_tokens} "
          f"completion={u.completion_tokens} cache_hit={hit}")
    return resp.choices[0].message.content


def ask_stream(client, question, model="deepseek-chat"):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        stream=True,
    )
    for chunk in stream:
        # the last chunk may carry only `usage` and an empty `choices` list
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def reasoner_dialogue(client, questions, model="deepseek-reasoner"):
    """Multi-turn chat with a reasoning model.

    The chain of thought comes back in `reasoning_content` and is billed as
    output tokens; only the final `content` goes back into the history, as
    the API documentation requires.
    """
    history = []
    for q in questions:
        history.append({"role": "user", "content": q})
        msg = client.chat.completions.create(
            model=model, messages=history).choices[0].message
        thinking = getattr(msg, "reasoning_content", "") or ""
        print(f"[reasoning: {len(thinking)} chars, billed as output]")
        history.append({"role": "assistant", "content": msg.content})
    return history[-1]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=DEEPSEEK_URL)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    client = make_client(a.base_url, timeout=a.timeout)
    print(ask(client, "What does MLA cache instead of full keys and values?"
              " One sentence.", a.model))
    ask_stream(client, "List three costs that an API price per token hides.",
               a.model)
    if a.base_url == DEEPSEEK_URL:
        print(reasoner_dialogue(client, ["Is 2**61 - 1 prime? Yes or no.",
                                         "And 2**67 - 1?"]))


if __name__ == "__main__":
    main()
