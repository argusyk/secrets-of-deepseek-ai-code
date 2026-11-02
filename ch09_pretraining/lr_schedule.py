"""The learning-rate and batch-size schedule of DeepSeek-V3 pre-training.

Chapter 9 of "Secrets of DeepSeek AI" (Listing 9.2). The constants follow
the DeepSeek-V3 technical report [1]: linear warm-up to 2.2e-4 over 2K steps, constant until
10T tokens, cosine decay to 2.2e-5 by 14.3T, then 2.2e-5 for 333B tokens and
7.3e-6 for the last 167B; the batch grows from 3072 to 15360 sequences of
4096 tokens during the first 469B tokens. The ramp shapes (linear batch growth
in tokens) are our simplifying assumption.

Run:
    python lr_schedule.py
"""
import math

SEQ_LEN = 4096
PEAK_LR, MID_LR, FINAL_LR = 2.2e-4, 2.2e-5, 7.3e-6
WARMUP_STEPS = 2000
T = 1e12                                   # one trillion tokens
CONST_END, COS_END = 10.0 * T, 14.3 * T
MID_END, TOTAL = 14.3 * T + 333e9, 14.8 * T
BATCH_START, BATCH_END, BATCH_RAMP = 3072, 15360, 469e9


def batch_size(tokens_seen):
    """Sequences per step; grows linearly (assumption) over the first 469B tokens."""
    frac = min(tokens_seen / BATCH_RAMP, 1.0)
    return int(round(BATCH_START + (BATCH_END - BATCH_START) * frac))


def learning_rate(step, tokens_seen):
    if step < WARMUP_STEPS:                              # linear warm-up
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    if tokens_seen < CONST_END:                          # constant phase
        return PEAK_LR
    if tokens_seen < COS_END:                            # cosine decay
        p = (tokens_seen - CONST_END) / (COS_END - CONST_END)
        cos = 0.5 * (1 + math.cos(math.pi * p))
        return MID_LR + (PEAK_LR - MID_LR) * cos
    if tokens_seen < MID_END:                            # 2.2e-5 for 333B tokens
        return MID_LR
    return FINAL_LR                                      # 7.3e-6 for 167B tokens


def simulate():
    """Walk the whole run step by step; return a list of (step, tokens, batch, lr)."""
    step, tokens, trace = 0, 0.0, []
    while tokens < TOTAL:
        b = batch_size(tokens)
        trace.append((step, tokens, b, learning_rate(step, tokens)))
        tokens += b * SEQ_LEN
        step += 1
    return trace


if __name__ == "__main__":
    trace = simulate()
    warm = trace[WARMUP_STEPS - 1]
    print(f"total optimizer steps : {len(trace):,}")
    print(f"tokens after warm-up  : {warm[1] / 1e9:.1f}B")
    print(f"tokens per step (max) : {BATCH_END * SEQ_LEN / 1e6:.1f}M")
    print("\n tokens(T)   batch    learning rate")
    marks = [0.001, 0.2, 0.469, 1, 5, 9.99, 11, 12.15, 13.5, 14.29, 14.5, 14.7]
    k = 0
    for step, tok, b, lr in trace:
        if k < len(marks) and tok >= marks[k] * T:
            print(f"{tok / T:9.3f}  {b:6d}   {lr:.2e}")
            k += 1
