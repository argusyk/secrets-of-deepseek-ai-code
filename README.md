# Secrets of DeepSeek AI — code examples

Runnable code examples accompanying the textbook **«Secrets of DeepSeek AI»**
(Ukrainian and English editions) by Ya.Yu. Dorohyi.

Each directory `chNN_<topic>` collects the listings printed in chapter NN of
the book; the files here are the complete versions of the (sometimes
shortened) listings in the text. The examples are deliberately small and
self-contained: each illustrates one idea (an MLA projection, a fine-grained
MoE router, the GRPO advantage, an API call) rather than reproducing the
production stack.

## Contents

| Directory | Chapter | What it shows | Mode |
|---|---|---|---|
| `ch02_attention` | 2 | Scaled dot-product and multi-head attention; parameter/FLOP accounting | run |
| `ch03_tokenization` | 3 | Toy BPE trainer; MinHash/LSH near-duplicate detection | run |
| `ch04_mla` | 4 | Multi-Head Latent Attention (naive vs absorbed); KV-cache sizes | run |
| `ch05_moe` | 5 | Fine-grained MoE layer with shared experts; V2/V3 routers | run |
| `ch06_balancing` | 6 | Auxiliary-loss-free (bias) balancing vs aux loss; MaxVio | run |
| `ch07_mtp` | 7 | Multi-token prediction loss; speculative decoding toy | run |
| `ch08_architecture` | 8 | Parameter and active-parameter count for V2/V3 | run |
| `ch09_pretraining` | 9 | V3 learning-rate/batch schedule; scaling-law fit | run |
| `ch10_fp8` | 10 | E4M3 emulation, tile/block scaling, accumulation error | run |
| `ch11_parallelism` | 11 | Pipeline bubbles (GPipe, 1F1B, ZB, DualPipe); 1F1B simulator | run |
| `ch12_post_training` | 12 | SFT loss masking, DPO loss, PPO clip | run |
| `ch13_grpo` | 13 | GRPO advantages, KL estimator, toy trainer; rule-based reward | run |
| `ch14_distillation` | 14 | Distillation loss with temperature; SFT-dataset builder | run |
| `ch15_inference` | 15 | KV-cache calculator; INT8/INT4 quantization error | run |
| `ch16_deployment` | 16 | Cost and memory calculators; API chat/streaming | run / API |
| `ch17_specialized` | 17 | Repo ordering, FIM prompts, pass@k, dynamic tiling; FIM API | run / API |
| `ch18_practice` | 18 | Code-review assistant with tools (--mock), retries, eval harness; JSON mode, reasoner | run / API |
| `ch19_safety` | 19 | Guardrail filter; prompt-injection test suite (mock model) | run |
| `ch20_future` | 20 | Jevons-effect demand simulation | run |

*run* – offline, NumPy only, no network or key needed.
*API* – calls the DeepSeek API and needs `DEEPSEEK_API_KEY`.

## Running

Python 3.10+.

```bash
pip install -r requirements.txt
python ch04_mla/mla_numpy.py
python ch13_grpo/grpo.py
python ch18_practice/code_review_assistant.py --mock ch18_practice/sample_buggy.py
```

The DeepSeek HTTP API is OpenAI-compatible (`base_url="https://api.deepseek.com"`).
For the API examples:

```bash
export DEEPSEEK_API_KEY="sk-..."
python ch16_deployment/api_chat.py
```

API parameters, model names and prices change; check the official
documentation at https://api-docs.deepseek.com/ before relying on them.

## License

The code is released under the MIT License (see `LICENSE`). The textbook text
itself is a separate work and is not covered by this license.
