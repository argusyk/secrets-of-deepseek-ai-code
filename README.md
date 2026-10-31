# Secrets of DeepSeek AI — code examples

Runnable code examples accompanying the textbook **«Secrets of DeepSeek AI»**
(Ukrainian and English editions) by Ya.Yu. Dorohyi.

Each directory `chNN_<topic>` collects the listings printed in the
corresponding chapter of the book. The examples are intentionally small and
self-contained: they illustrate one idea at a time (an MLA projection, a
fine-grained MoE router, the GRPO advantage, an API call) rather than
reproducing the full production stack.

> The textbook refers to this repository as
> `https://github.com/argusyk/secrets-of-deepseek-ai-code`.

## Layout

```
chNN_<topic>/     # listings for chapter NN
requirements.txt  # Python dependencies for the runnable examples
```

## Running

Most listings target Python 3.10+. Install the dependencies:

```bash
pip install -r requirements.txt
```

The API examples use the DeepSeek platform, whose HTTP API is
OpenAI-compatible. Export your key before running them:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

The illustrative numerical listings (attention, MoE routing, GRPO) depend only
on `numpy`/`torch` and run offline, without network access or an API key.

## License

The code is released under the MIT License (see `LICENSE`). The textbook text
itself is a separate work and is not covered by this license.
