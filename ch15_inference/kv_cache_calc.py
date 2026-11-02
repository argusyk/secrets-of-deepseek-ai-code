"""KV-cache capacity planning: cache per token and Little's law.

Teaching calculator for Chapter 15 of "Secrets of DeepSeek AI". The cache
formulas themselves are those of Chapter 4 ((4.1), (4.10), Table 4.1); this
script answers the serving questions of Section 15.6: how many sequences are
in the system at once (Little's law, formula (15.7)) and how much memory
their caches need. The MLA row uses DeepSeek-V3's latent (512) plus
decoupled RoPE key (64) per layer; the other rows reuse V3's layer count and
head geometry so the comparison isolates the attention scheme.

Units: cache sizes are printed in binary units (KiB, GiB = 2**30 bytes) as in
Chapter 4; memory budgets of accelerators are decimal GB (10**9 bytes).

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


def in_flight(rate, seconds):
    """Little's law (15.7): mean number of requests in the system."""
    return rate * seconds


def cache_for_traffic(kind, rate, seconds, context, **kw):
    """Bytes of KV cache for the sequences that are in flight at once."""
    return cache_bytes(kind, context, batch=in_flight(rate, seconds), **kw)


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:7.1f} {unit}"
        n /= 1024
    return f"{n:7.1f} PiB"


if __name__ == "__main__":
    kinds = ["MHA", "GQA", "MQA", "MLA"]
    print("Cache per token (all 61 layers, BF16):")
    for k in kinds:
        print(f"  {k:4s} {human(bytes_per_token(k))}")

    budget = 400  # GB of accelerator memory left for the cache (decimal)
    print(f"\nConcurrent 128K-token sessions in a {budget} GB cache budget:")
    for k in kinds:
        print(f"  {k:4s} {max_sequences(k, 131_072, budget):5d}")

    rate, seconds, ctx = 20, 15, 8_192
    n = in_flight(rate, seconds)
    print(f"\nLittle's law: {rate} req/s x {seconds} s = {n:.0f} sequences"
          f" of {ctx} tokens in flight")
    for k in ("GQA", "MLA"):
        b = cache_for_traffic(k, rate, seconds, ctx)
        print(f"  {k:4s} {human(b)} = {b / 1e9:6.1f} GB")

    ratio = bytes_per_token("MHA") / bytes_per_token("MLA")
    print(f"\nMHA / MLA cache ratio: {ratio:.1f}x")
