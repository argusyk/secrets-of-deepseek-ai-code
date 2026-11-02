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
| `tinyseek` | 2–18 | The TinySeek package: a miniature DeepSeek-V3 in PyTorch (see below) | run |
| `tinyseek_labs` | App. C | Eight laboratory scripts | run |
| `tests` | – | pytest suite for TinySeek and the chapter examples | run |

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

## TinySeek: a miniature DeepSeek-V3 you can train on a laptop

`tinyseek/` is the through-line project of the book: a DeepSeek-V3-style
model in PyTorch that runs on a CPU in minutes. Each chapter adds one idea
(the "TinySeek step" sections), and the eight labs of Appendix C
(`tinyseek_labs/`) turn them into measurements.

The `tiny_v3` preset: vocabulary 2048, width 128, 4 layers (the first one
dense), 4 heads, MLA (d_c = 32, d_c' = 48, d_h = 32, d_h^R = 16), DeepSeekMoE
with 1 shared + 8 routed experts of width 64 (top-2, sigmoid router,
auxiliary-loss-free bias balancing), one MTP module, context 256 –
1.48M parameters (1.03M active per token, 0.30M more in the MTP module).

### Installation

```bash
cd code
pip install -r requirements.txt          # torch (CPU wheel is enough), numpy, pytest
python -m pytest -q                      # ~100 tests, about 40 s on a free CPU
```

### Quick start

```bash
python -m tinyseek.data prepare                         # corpus + BPE + tokens, ~10 s
python -m tinyseek.train --preset tiny_v3 --steps 1500 --out runs/v3
python -m tinyseek.sft --init runs/v3 --steps 400 --out runs/sft
python -m tinyseek.grpo --init runs/sft --steps 40 --out runs/grpo
python -m tinyseek.generate --run runs/sft --prompt "Q: 47+38=?\nA: " --speculative
python -m tinyseek.serve --run runs/sft --port 8000      # OpenAI-compatible API
```

Every command accepts `--help` and `--seed` (default 0). The corpus is built
offline: source files of the Python standard library of your interpreter,
Ukrainian and English prose in `tinyseek/data_texts/` (the prose of this
textbook with formulas, code and markup removed, ~0.9 MB) and synthetic
arithmetic problems with a column-wise chain of thought. Because the code
part comes from your Python installation, numbers can differ slightly
between Python versions.

### Modules

| Module | What it contains | Chapter |
|---|---|---|
| `tinyseek.config` | `TinySeekConfig` with DeepSeek `config.json` field names; presets `tiny_dense`, `tiny_mla`, `tiny_moe`, `tiny_v3`; `to_json` / `from_json`; analytic parameter count (`configs/deepseek_v3.json` gives 671B / 37B) | 2, 8 |
| `tinyseek.tokenizer` | `ByteBPETokenizer` (`train`, `encode`, `decode`, `save`, `load`, `fertility`), special tokens, `fim_transform` (PSM) | 3, 17 |
| `tinyseek.data` | `build_corpus`, `minhash_dedup`, `PackedDataset`, `make_arith_task`, `prepare_data` | 3, 12–14 |
| `tinyseek.model` | `RMSNorm`, `apply_rope`, `MultiHeadAttention`, `MultiHeadLatentAttention` (latent cache, `absorbed=True`), `SwiGLU`, `MoE` (`update_bias`, `last_load`, `aux_loss`), `MTPModule`, `TinySeekModel`, `count_parameters`, `kv_cache_bytes` | 2, 4–8 |
| `tinyseek.train` | `train`, `lr_schedule` (V3 shape), JSONL log with loss, perplexity, MaxVio; checkpoints | 6, 9 |
| `tinyseek.fp8` | `fake_quant_e4m3` (1×128 tiles, 128×128 blocks), `FP8Linear`, `convert_to_fp8` | 10 |
| `tinyseek.parallel_sim` | event simulation of 1F1B and DualPipe on virtual accelerators, measured F/B/W, all-to-all traffic | 11 |
| `tinyseek.sft` | SFT with prompt masking on arithmetic instructions | 12 |
| `tinyseek.grpo` | `grpo_advantages`, `kl_k3`, `grpo_loss`, `rule_reward`, `grpo_train` | 13 |
| `tinyseek.distill` | rejection-sampled teacher traces, `kd_loss` with temperature, students | 14 |
| `tinyseek.generate` | `sample`, cached generation, `speculative_generate` with the MTP draft, `quantize_int8` | 2, 7, 15 |
| `tinyseek.serve` | OpenAI-compatible `/v1/chat/completions` on `http.server` (with streaming) | 16, 18 |
| `tinyseek.evaluate` | perplexity, arithmetic accuracy, FIM pass@k | 9, 13, 17 |

### Labs (Appendix C)

Run from `code/`; each lab prints a results table and writes CSV and SVG
files to `runs/labN/`. Times are for a 4-core laptop CPU.

| Lab | Topic | Chapters | Script | Time |
|---|---|---|---|---|
| 1 | Corpus and tokenizer: BPE, fertility, FIM, deduplication | 2–3 | `python tinyseek_labs/lab1_tokenizer.py` | ~0.5 min |
| 2 | From MHA to MLA: KV cache and matrix absorption | 2, 4 | `python tinyseek_labs/lab2_mla.py` | ~3 min |
| 3 | Mixture of experts and bias balancing | 5–6 | `python tinyseek_labs/lab3_moe.py` | ~6 min |
| 4 | MTP and training TinySeek-V3 with the V3 schedule | 7–9 | `python tinyseek_labs/lab4_train.py` | ~7 min (14 on a busy CPU) |
| 5 | FP8 emulation and pipeline parallelism | 10–11 | `python tinyseek_labs/lab5_efficiency.py` | ~6 min (10 on a busy CPU) |
| 6 | SFT and GRPO on a verifiable task | 12–13 | `python tinyseek_labs/lab6_rl.py` | ~7 min |
| 7 | Distilling reasoning into a smaller model | 14 | `python tinyseek_labs/lab7_distill.py` | ~4 min |
| 8 | Cache, INT8, speculative decoding, OpenAI-compatible API | 15–18 | `python tinyseek_labs/lab8_serve.py` | ~1 min |

Labs 5–8 reuse the checkpoints of labs 4 and 6 when they exist and
otherwise train small ones themselves, so any lab can be run on its own.
Several TinySeek processes on one CPU slow each other down a lot; the
package sets `OMP_WAIT_POLICY=PASSIVE` (unless already set) to soften this,
but for timing measurements run one job at a time.

### Tests

`tests/` checks tensor shapes, the equality of naive and absorbed MLA, the
active-parameter count of MoE, the drop of MaxVio under bias balancing, the
MTP loss, tokenizer and FIM round trips, cached vs uncached logits, the
statistics of GRPO advantages, speculative decoding (greedy equality and the
acceptance rule), FP8, the pipeline simulator, the server, and runs every
offline example in `chNN_*/` (API examples are skipped).

## License

The code is released under the MIT License (see `LICENSE`). The textbook text
itself is a separate work and is not covered by this license.
