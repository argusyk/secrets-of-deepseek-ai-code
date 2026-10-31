"""Approximate parameter counts for DeepSeek-V2 and DeepSeek-V3.

Chapter 8 of "Secrets of DeepSeek AI". The script rebuilds the total and the
per-token (active) parameter counts from the published configuration values,
layer by layer, and also reports the KV-cache footprint of MLA versus a
conventional multi-head attention with the same number of heads.

The estimate ignores tiny tensors (biases, a few norm vectors are included,
router biases are not) and the Multi-Token Prediction module, which is
shipped separately in the released checkpoint. It is an approximation, not a
dump of the real weights.

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
    """Multi-head Latent Attention: down/up projections and output."""
    q_down = c.d * c.q_rank + c.q_rank                       # W_DQ + norm
    q_up = c.q_rank * c.heads * (c.d_nope + c.d_rope)        # W_UQ (+ W_QR)
    kv_down = c.d * (c.kv_rank + c.d_rope) + c.kv_rank       # W_DKV, W_KR + norm
    kv_up = c.kv_rank * c.heads * (c.d_nope + c.d_v)         # W_UK, W_UV
    out = c.heads * c.d_v * c.d                              # W_O
    return q_down + q_up + kv_down + kv_up + out


def swiglu(d, inner):
    """Gate, up and down projections of a SwiGLU feed-forward network."""
    return 3 * d * inner


def count(c):
    attn = mla_params(c)
    norms = 2 * c.d
    dense = swiglu(c.d, c.dense_ffn)
    expert = swiglu(c.d, c.expert_ffn)
    router = c.n_routed * c.d
    moe_layers = c.layers - c.dense_layers

    total = c.layers * (attn + norms)
    total += c.dense_layers * dense
    total += moe_layers * ((c.n_routed + c.n_shared) * expert + router)
    embed = c.vocab * c.d
    total += 2 * embed + c.d                                 # embedding, head, final norm

    active = c.layers * (attn + norms)
    active += c.dense_layers * dense
    active += moe_layers * ((c.top_k + c.n_shared) * expert + router)
    active += 2 * embed + c.d

    experts_share = moe_layers * c.n_routed * expert / total
    return total, active, experts_share


def kv_cache_per_token(c):
    """Elements cached per token: MLA latent + RoPE key vs full MHA K and V.

    The MHA baseline keeps a key and a value of d_v = 128 per head, i.e. a
    conventional head size with the same number of heads.
    """
    mla = c.layers * (c.kv_rank + c.d_rope)
    mha = c.layers * 2 * c.heads * c.d_v
    return mla, mha


if __name__ == "__main__":
    for c in (V3, V2):
        total, active, share = count(c)
        mla, mha = kv_cache_per_token(c)
        print(f"{c.name}")
        print(f"  total params   : {total / 1e9:7.1f} B")
        print(f"  active / token : {active / 1e9:7.1f} B  ({active / total:.1%})")
        print(f"  routed experts : {share:.1%} of all parameters")
        print(f"  KV cache/token : MLA {mla:,} vs MHA {mha:,} elements "
              f"(x{mha / mla:.0f} smaller); BF16: {mla * 2 / 1024:.0f} KB "
              f"vs {mha * 2 / 1024 / 1024:.1f} MB")
