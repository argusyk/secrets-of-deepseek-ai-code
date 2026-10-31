"""How much does a transformer cost? Parameters, FLOPs and arithmetic intensity.

Back-of-the-envelope estimates for a standard decoder block (multi-head
attention + SwiGLU feed-forward network), as used in Chapter 2 of
"Secrets of DeepSeek AI". The worked example uses the width of DeepSeek-V3
(d_model = 7168, dense FFN width 18432, vocabulary 129280) but a plain MHA
attention; DeepSeek's latent attention (MLA) is treated in Chapter 4.

Rules of thumb used here:
  * forward pass  ~ 2 * N FLOPs per token (N = parameters touched per token);
  * training step ~ 6 * N FLOPs per token (forward + backward ~ 2x forward);
  * attention scores add ~ 4 * ctx * d FLOPs per token per layer (forward);
  * a matrix-vector product reads every weight once, so at batch size B the
    arithmetic intensity is ~ B FLOP per byte for 2-byte (BF16) weights.

Run:
    python flops_params.py
"""


def block_params(d_model, d_ff):
    """Parameters of one block: MHA (Q, K, V, O) + SwiGLU FFN (W1, W3, W2) + 2 RMSNorm gains."""
    attn = 4 * d_model * d_model
    ffn = 3 * d_model * d_ff
    norms = 2 * d_model
    return {"attention": attn, "ffn": ffn, "norms": norms, "total": attn + ffn + norms}


def forward_flops_per_token(n_params, d_model=None, ctx=0, n_layers=1):
    """~2 FLOPs per parameter, plus the attention-score term for a context of ctx tokens."""
    flops = 2 * n_params
    if ctx and d_model:
        flops += 4 * ctx * d_model * n_layers      # Q K^T and (weights) V, 2 FLOPs per MAC each
    return flops


def ridge_point(peak_flops, mem_bandwidth):
    """Arithmetic intensity (FLOP/byte) above which a kernel becomes compute-bound."""
    return peak_flops / mem_bandwidth


def decode_time_lower_bound(n_params, bytes_per_param, mem_bandwidth):
    """At batch size 1 every weight is read once per generated token."""
    return n_params * bytes_per_param / mem_bandwidth


def fmt(x):
    for unit, k in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= k:
            return f"{x / k:.2f}{unit}"
    return f"{x:.0f}"


if __name__ == "__main__":
    d, d_ff, vocab, ctx = 7168, 18432, 129280, 4096
    p = block_params(d, d_ff)
    print("One block, d_model=7168, d_ff=18432 (plain MHA + SwiGLU):")
    for k in ("attention", "ffn", "norms", "total"):
        print(f"  {k:9s} {p[k]:>13,d}  ({fmt(p[k])})")
    print(f"  FFN share of the block: {p['ffn'] / p['total']:.1%}")

    f_lin = forward_flops_per_token(p["total"])
    f_att = 4 * ctx * d
    print(f"\nForward FLOPs per token per block: {fmt(f_lin)} (linear layers)"
          f" + {fmt(f_att)} (attention scores at ctx={ctx})")
    print(f"Training FLOPs per token per block (~6N): {fmt(6 * p['total'])}")

    emb = vocab * d
    print(f"\nEmbedding table: {fmt(emb)} parameters; the output projection costs"
          f" {fmt(2 * emb)} FLOPs per token")

    # an accelerator of the H100/H800 class: ~1e15 FLOP/s dense BF16, ~3.35e12 B/s HBM
    peak, bw = 989e12, 3.35e12
    print(f"\nRidge point: {ridge_point(peak, bw):.0f} FLOP/byte"
          " -> decoding with batch < ~300 tokens is memory-bound")
    for n in (7e9, 37e9, 70e9):
        t = decode_time_lower_bound(n, 2, bw)
        print(f"  {fmt(n)} parameters read per token in BF16: >= {t * 1e3:.1f} ms/token"
              f" (<= {1 / t:.0f} tokens/s at batch 1)")
