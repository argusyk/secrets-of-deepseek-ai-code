"""KV-cache size for MHA, GQA, MQA and MLA with DeepSeek-V3-like dimensions.

Chapter 4 of "Secrets of DeepSeek AI". The MHA/GQA/MQA rows are hypothetical
models with the same number of layers, heads and head dimension; only the
attention variant (and therefore what is cached) changes.

Run:
    python kv_cache_size.py
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


def report(context=128 * 1024):
    base = per_token_per_layer("MHA")
    print(f"{'variant':8}{'elems/tok/layer':>17}{'KiB/token':>12}"
          f"{'GiB @ ' + str(context // 1024) + 'K':>14}{'% of MHA':>10}")
    for v, label in (("MHA", "MHA"), ("GQA", "GQA-8"), ("MQA", "MQA"), ("MLA", "MLA")):
        e = per_token_per_layer(v)
        per_tok_bytes = e * N_LAYERS * BYTES
        total = per_tok_bytes * context
        print(f"{label:8}{e:>17}{per_tok_bytes / 1024:>12.1f}"
              f"{total / 2**30:>14.2f}{100 * e / base:>10.2f}")
    print(f"MHA / MLA ratio: {base / per_token_per_layer('MLA'):.1f}x")
    # MLA cache expressed as an equivalent number of GQA groups
    print(f"MLA equals GQA with {(D_C + D_HR) / (2 * D_HEAD):.2f} groups")


if __name__ == "__main__":
    report()
