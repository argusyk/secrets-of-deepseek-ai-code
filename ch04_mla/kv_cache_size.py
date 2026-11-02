"""KV-cache size for MHA, GQA, MQA and MLA with DeepSeek-V3-like dimensions.

Chapter 4 of "Secrets of DeepSeek AI". The MHA/GQA/MQA rows are hypothetical
models with the same number of layers, heads and head dimension; only the
attention variant (and therefore what is cached) changes.

GiB = 2**30 bytes (binary), GB = 10**9 bytes (decimal, as in GPU
specifications): 488 GiB and 524 GB are the same amount of memory.

Run:
    python kv_cache_size.py          # 128K-token context
    python kv_cache_size.py 32       # 32K-token context
"""

N_LAYERS = 61        # DeepSeek-V3
N_HEADS = 128
D_HEAD = 128
D_C = 512            # MLA latent dimension (kv_lora_rank)
D_HR = 64            # decoupled RoPE key dimension
BYTES = 2            # BF16


def per_token_per_layer(variant, n_groups=8):
    if variant == "MHA":
        return 2 * N_HEADS * D_HEAD
    if variant == "GQA":
        return 2 * n_groups * D_HEAD
    if variant == "MQA":
        return 2 * D_HEAD
    if variant == "MLA":
        return D_C + D_HR
    raise ValueError(variant)


def kv_cache_bytes(variant, seq_len, n_layers=N_LAYERS, n_groups=8):
    """Bytes of KV cache for one sequence of seq_len tokens."""
    per_tok = per_token_per_layer(variant, n_groups) * n_layers
    return per_tok * seq_len * BYTES


def report(context=128 * 1024, budget_gb=400):
    base = per_token_per_layer("MHA")
    print(f"{'variant':8}{'elems/tok/layer':>17}{'KiB/token':>12}"
          f"{'GiB @ ' + str(context // 1024) + 'K':>14}"
          f"{'GB @ ' + str(context // 1024) + 'K':>13}{'% of MHA':>10}"
          f"{'sessions':>10}")
    for v, label in (("MHA", "MHA"), ("GQA", "GQA-8"), ("MQA", "MQA"),
                     ("MLA", "MLA")):
        e = per_token_per_layer(v)
        total = kv_cache_bytes(v, context)
        per_tok_bytes = total // context
        sessions = int(budget_gb * 1e9 // total)
        print(f"{label:8}{e:>17}{per_tok_bytes / 1024:>12.1f}"
              f"{total / 2**30:>14.2f}{total / 1e9:>13.1f}"
              f"{100 * e / base:>10.2f}{sessions:>10}")
    print(f"(sessions = how many {context // 1024}K-token sequences fit "
          f"into {budget_gb} GB = {budget_gb}e9 bytes of cache)")
    print(f"MHA / MLA ratio: {base / per_token_per_layer('MLA'):.1f}x")
    # MLA cache expressed as an equivalent number of GQA groups
    print(f"MLA equals GQA with {(D_C + D_HR) / (2 * D_HEAD):.2f} groups")


if __name__ == "__main__":
    import sys
    ctx = int(sys.argv[1]) * 1024 if len(sys.argv) > 1 else 128 * 1024
    report(ctx)
