"""When does speculative decoding pay off on a server? (Section 15.5)

Teaching model for Chapter 15 of "Secrets of DeepSeek AI". One forward pass
over `batch` sequences with `m` new tokens each takes the roofline time of
formulas (15.2)-(15.3): reading the weights and the KV caches, or doing the
arithmetic, whichever is slower. Speculative decoding with a k-token draft
replaces one plain step by a verification pass over k+1 tokens per sequence
plus the cost of drafting; it yields E[N] tokens (formula (7.11)).

    S(B) = E[N] * t(B, 1) / (t(B, k+1) + t_draft)        (formula (15.6))

The defaults describe a dense 7B model in BF16 with a GQA cache on one
accelerator of the H800 class; they are illustrative round numbers.

Run:
    python spec_batch.py
"""

HW = dict(
    weight_bytes=14e9,         # 7B parameters x 2 bytes (BF16)
    flops_per_token=14e9,      # about 2 N FLOPs per token
    kv_bytes_per_token=57_344,  # 28 layers x 4 KV heads x 128 x 2 x 2 bytes
    peak_flops=989e12,         # dense BF16 peak
    bandwidth=3.35e12,         # HBM bytes per second
)


def step_time(batch, m=1, context=2048, weight_bytes=14e9,
              flops_per_token=14e9, kv_bytes_per_token=57_344,
              peak_flops=989e12, bandwidth=3.35e12):
    """Roofline time of one pass: batch sequences, m new tokens each."""
    memory = weight_bytes + batch * context * kv_bytes_per_token
    compute = batch * m * flops_per_token
    return max(memory / bandwidth, compute / peak_flops)


def expected_tokens(alpha, k):
    """Formula (7.11): tokens per verification pass of the target model."""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def spec_speedup(batch, k=1, alpha=0.85, draft_cost=0.05, **kw):
    """Formula (15.6); draft_cost = one draft token / one plain step."""
    plain = step_time(batch, 1, **kw)
    spec = step_time(batch, k + 1, **kw) + k * draft_cost * plain
    return expected_tokens(alpha, k) * plain / spec


def fits(batch, context, memory=80e9, weight_bytes=14e9,
         kv_bytes_per_token=57_344, **_):
    """Do the weights and all KV caches fit into one accelerator?"""
    return weight_bytes + batch * context * kv_bytes_per_token <= memory


if __name__ == "__main__":
    ridge = HW["peak_flops"] / HW["bandwidth"]
    print(f"ridge point of the accelerator: {ridge:.0f} FLOP/byte")
    print(f"weights alone: {HW['weight_bytes'] / HW['bandwidth'] * 1e3:.1f} ms"
          " per decode step\n")
    cases = [(512, 1, 0.85), (512, 3, 0.7), (4096, 1, 0.85), (4096, 3, 0.7)]
    head = "".join(f"  ctx {c}, k={k}" for c, k, _ in cases)
    print(f"{'batch':>6s}{head}")
    for b in (1, 32, 128, 256, 512, 1024):
        row = ""
        for ctx, k, a in cases:
            ok = fits(b, ctx)
            s = spec_speedup(b, k=k, alpha=a, context=ctx) if ok else None
            row += f"{s:14.2f}" if ok else f"{'-':>14s}"
        print(f"{b:6d}{row}")
    print("\nk=1: MTP-like draft, alpha=0.85; k=3: separate draft, alpha=0.7;"
          "\n'-': the KV caches do not fit into 80 GB next to the weights")
