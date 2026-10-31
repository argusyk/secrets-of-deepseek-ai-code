"""Rough memory estimate for serving a model locally.

Chapter 16 of "Secrets of DeepSeek AI". The estimate is deliberately simple:

    total = weights + kv_cache + overhead
    weights  = n_params * bytes_per_param
    kv_cache = tokens_in_flight * kv_bytes_per_token
    overhead = a fraction of the weights (activations, CUDA context, fragmentation)

It is an order-of-magnitude planning tool, not a guarantee: real engines differ
in how they quantize, shard and pre-allocate memory.

Run:
    python local_requirements.py
"""
import math

BYTES = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


def kv_bytes_per_token_mla(n_layers=61, d_c=512, d_rope=64, dtype_bytes=2):
    """MLA caches one latent vector plus one shared RoPE key per layer."""
    return n_layers * (d_c + d_rope) * dtype_bytes


def kv_bytes_per_token_gqa(n_layers, n_kv_heads, d_head, dtype_bytes=2):
    """Grouped-query attention caches K and V for each KV head."""
    return 2 * n_layers * n_kv_heads * d_head * dtype_bytes


def estimate(n_params, precision, kv_per_token, tokens_in_flight, overhead=0.10):
    w = n_params * BYTES[precision]
    kv = kv_per_token * tokens_in_flight
    total = w * (1 + overhead) + kv
    return w, kv, total


def gpus_needed(total_bytes, gpu_gb):
    return math.ceil(total_bytes / (gpu_gb * 1e9))


def show(name, n_params, precision, kv_per_token, tokens):
    w, kv, total = estimate(n_params, precision, kv_per_token, tokens)
    print(f"{name:<34} {precision:>5}  weights {w/1e9:7.0f} GB  "
          f"KV {kv/1e9:6.1f} GB  total {total/1e9:7.0f} GB  "
          f"H100-80GB x{gpus_needed(total, 80):<3} H200-141GB x{gpus_needed(total, 141)}")


if __name__ == "__main__":
    mla = kv_bytes_per_token_mla()
    print(f"MLA KV cache per token (V3, BF16): {mla/1024:.1f} KiB\n")
    tokens = 32 * 8192          # e.g. 32 concurrent requests x 8K context
    print(f"tokens kept in KV cache: {tokens}\n")
    show("DeepSeek-V3/R1 671B", 671e9, "fp8", mla, tokens)
    show("DeepSeek-V3/R1 671B", 671e9, "bf16", mla, tokens)
    show("DeepSeek-V3/R1 671B (4-bit quant)", 671e9, "int4", mla, tokens)
    # distilled dense models use the base model's GQA layout; values are typical,
    # not exact, and only illustrate the order of magnitude
    show("R1-Distill 70B (Llama-like GQA)", 70e9, "bf16",
         kv_bytes_per_token_gqa(80, 8, 128), tokens)
    show("R1-Distill 32B (Qwen-like GQA)", 32e9, "int4",
         kv_bytes_per_token_gqa(64, 8, 128), 4 * 8192)
    show("R1-Distill 7B (Qwen-like GQA)", 7e9, "bf16",
         kv_bytes_per_token_gqa(28, 4, 128), 4 * 8192)
