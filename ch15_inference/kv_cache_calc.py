"""KV-cache memory for MHA, GQA, MQA and MLA as a function of context length.

Teaching calculator for Chapter 15 of "Secrets of DeepSeek AI". It answers two
practical questions: how many bytes one token of context costs in the cache,
and how many concurrent sequences of a given length fit into a memory budget.
The MLA row uses DeepSeek-V3's latent (512) plus decoupled RoPE key (64)
per layer; the other rows reuse V3's layer count and head geometry so the
comparison isolates the attention scheme.

Run:
    python kv_cache_calc.py
"""

LAYERS = 61          # DeepSeek-V3
HEADS = 128
HEAD_DIM = 128
D_LATENT = 512       # kv_lora_rank
D_ROPE = 64          # qk_rope_head_dim
BYTES = 2            # BF16 cache entries


def elems_per_token_per_layer(kind, kv_groups=8):
    """Number of cached scalars for one token in one layer."""
    if kind == "MHA":
        return 2 * HEADS * HEAD_DIM            # keys + values for every head
    if kind == "GQA":
        return 2 * kv_groups * HEAD_DIM        # keys + values per group
    if kind == "MQA":
        return 2 * HEAD_DIM                    # one shared key/value head
    if kind == "MLA":
        return D_LATENT + D_ROPE               # latent c_KV + shared RoPE key
    raise ValueError(kind)


def bytes_per_token(kind, **kw):
    return elems_per_token_per_layer(kind, **kw) * LAYERS * BYTES


def cache_bytes(kind, context, batch=1, **kw):
    return bytes_per_token(kind, **kw) * context * batch


def max_sequences(kind, context, budget_gb, **kw):
    """How many sequences of `context` tokens fit into `budget_gb` of memory."""
    return int(budget_gb * 1e9 // cache_bytes(kind, context, **kw))


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000:
            return f"{n:7.1f} {unit}"
        n /= 1000
    return f"{n:7.1f} PB"


if __name__ == "__main__":
    kinds = ["MHA", "GQA", "MQA", "MLA"]
    print("Cache per token (all 61 layers, BF16):")
    for k in kinds:
        print(f"  {k:4s} {human(bytes_per_token(k))}")

    print("\nCache for one sequence:")
    print("  ctx      " + "".join(f"{k:>12s}" for k in kinds))
    for ctx in (4_096, 32_768, 131_072):
        row = "".join(f"{human(cache_bytes(k, ctx)):>12s}" for k in kinds)
        print(f"  {ctx:7d}  {row}")

    budget = 400  # GB of accelerator memory left for the cache
    print(f"\nConcurrent 128K-token sessions in a {budget} GB cache budget:")
    for k in kinds:
        print(f"  {k:4s} {max_sequences(k, 131_072, budget):5d}")

    ratio = bytes_per_token("MHA") / bytes_per_token("MLA")
    print(f"\nMHA / MLA cache ratio: {ratio:.1f}x")
