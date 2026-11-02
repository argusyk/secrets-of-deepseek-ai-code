"""Approximate parameter counts for DeepSeek-V2 and DeepSeek-V3.

Chapter 8 of "Secrets of DeepSeek AI" (Listing 8.1). The script rebuilds the
total and the per-token (active) parameter counts of the main model from the
published configuration values, layer by layer, and also reports the
KV-cache footprint of MLA versus a conventional multi-head attention with
the same number of heads (binary units: KiB, MiB).

The estimate ignores tiny tensors (biases, the routers' balancing biases)
and the Multi-Token Prediction module: the 671B of DeepSeek-V3 is the main
model only, while the released checkpoint (685B) also holds the MTP module
together with its own copies of the embedding and the output head. It is an
approximation, not a dump of the real weights. The same formulas are used
by `tinyseek.model.count_parameters` for the tiny `tiny_v3` preset.

Run:
    python param_count.py
"""
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    d: int               # hidden size
    layers: int          # transformer blocks of the main model
    heads: int           # attention heads
    d_nope: int          # per-head query/key dim without RoPE
    d_rope: int          # per-head decoupled RoPE dim
    d_v: int             # per-head value dim
    q_rank: int          # query compression dim (d_c')
    kv_rank: int         # KV latent dim (d_c)
    dense_layers: int    # first layers that keep a dense FFN
    dense_ffn: int       # intermediate size of the dense FFN
    n_routed: int        # routed experts per MoE layer
    n_shared: int        # shared experts per MoE layer
    top_k: int           # routed experts activated per token
    expert_ffn: int      # intermediate size of one expert
    vocab: int


V3 = Config("DeepSeek-V3", 7168, 61, 128, 128, 64, 128, 1536, 512,
            3, 18432, 256, 1, 8, 2048, 129280)
V2 = Config("DeepSeek-V2", 5120, 60, 128, 128, 64, 128, 1536, 512,
            1, 12288, 160, 2, 6, 1536, 102400)


def mla_params(c):
    """Multi-head Latent Attention of one block."""
    q_down = c.d * c.q_rank + c.q_rank                 # W_DQ + norm
    q_up = c.q_rank * c.heads * (c.d_nope + c.d_rope)  # W_UQ, W_QR
    kv_down = c.d * (c.kv_rank + c.d_rope) + c.kv_rank # W_DKV, W_KR
    kv_up = c.kv_rank * c.heads * (c.d_nope + c.d_v)   # W_UK, W_UV
    out = c.heads * c.d_v * c.d                        # W_O
    return q_down + q_up + kv_down + kv_up + out


def count(c):
    """Total and active parameters of the main model (no MTP module)."""
    attn, expert = mla_params(c), 3 * c.d * c.expert_ffn   # SwiGLU
    moe_layers = c.layers - c.dense_layers
    base = (c.layers * (attn + 2 * c.d)                    # + 2 norms
            + c.dense_layers * 3 * c.d * c.dense_ffn
            + moe_layers * c.n_routed * c.d                # routers
            + 2 * c.vocab * c.d + c.d)                     # emb, head
    total = base + moe_layers * (c.n_routed + c.n_shared) * expert
    active = base + moe_layers * (c.top_k + c.n_shared) * expert
    return total, active


def routed_share(c):
    """Share of all parameters that sits in the routed experts."""
    total, _ = count(c)
    moe_layers = c.layers - c.dense_layers
    return moe_layers * c.n_routed * 3 * c.d * c.expert_ffn / total


def kv_cache_per_token(c):
    """Elements cached per token: MLA latent + RoPE key vs full MHA K and V.

    The MHA baseline keeps a key and a value of d_v = 128 per head, i.e. a
    conventional head size with the same number of heads.
    """
    mla = c.layers * (c.kv_rank + c.d_rope)
    mha = c.layers * 2 * c.heads * c.d_v
    return mla, mha


def human_bytes(n):
    """Binary units, as in Table 4.1: KiB = 1024 B, MiB = 1024 KiB."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


if __name__ == "__main__":
    for c in (V3, V2):
        total, active = count(c)
        mla, mha = kv_cache_per_token(c)
        print(f"{c.name}")
        print(f"  MLA per block  : {mla_params(c) / 1e6:7.1f} M")
        print(f"  total params   : {total / 1e9:7.1f} B")
        print(f"  active / token : {active / 1e9:7.1f} B"
              f"  ({active / total:.1%})")
        print(f"  routed experts : {routed_share(c):.1%} of all parameters")
        print(f"  KV cache/token : MLA {mla:,} vs MHA {mha:,} elements "
              f"(x{mha / mla:.0f} smaller)")
        print(f"  BF16 per token : {human_bytes(2 * mla)} "
              f"vs {human_bytes(2 * mha)}")
