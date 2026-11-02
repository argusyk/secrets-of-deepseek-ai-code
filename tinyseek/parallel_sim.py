"""Pipeline schedules and all-to-all traffic on "virtual accelerators" (Chapter 11).

TinySeek fits on one CPU, so parallelism is simulated: the model is split
into ``p`` equal pipeline stages whose forward (F), backward (B) and
weight-gradient (W) times can be measured on the real TinySeek blocks
(`measure_stage_times`). Two schedules are then executed by the same
event-driven simulator with per-device operation orders:

* `schedule_1f1b` - classic one-forward-one-backward;
* `schedule_dualpipe` - the bidirectional DualPipe schedule of DeepSeek-V3
  (eight phases per device, overlapped forward+backward chunk pairs, and
  zero-bubble splitting of B into input-gradient and W parts).

Communication (the expert-parallel all-to-all) costs ``comm`` time units per
chunk; in DualPipe an overlapped F+B pair hides it (formula (11.3)). The
bubble formulas of the V3 report are in `bubble_formula` (11.4).

`all_to_all_bytes` counts dispatch/combine traffic for actual TinySeek
routing decisions when the routed experts are spread over ``ep`` devices,
with and without V3-style deduplication of tokens per destination device.

CLI::

    python -m tinyseek.parallel_sim --p 4 --m 16
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import torch

Key = tuple  # (kind, direction, microbatch, stage)


# --------------------------------------------------------------------- formulas
def bubble_formula(kind: str, p: int, F: float, B: float, W: float = 0.0,
                   FB: Optional[float] = None) -> float:
    """Bubble length per the DeepSeek-V3 report (Table 2; formula (11.4))."""
    if kind == "1F1B":
        return (p - 1) * (F + B)
    if kind == "ZB1P":
        return (p - 1) * (F + B - 2 * W)
    if kind == "DualPipe":
        FB = F + B if FB is None else FB
        return (p / 2 - 1) * (FB + B - 3 * W)
    raise ValueError(kind)


# ---------------------------------------------------------------- simulator
def _device(direction: int, stage: int, p: int) -> int:
    return stage if direction == 0 else p - 1 - stage


def simulate(orders: list[list[tuple]], p: int, F: float, B: float, W: float,
             comm: float = 0.0) -> dict:
    """Execute per-device operation lists respecting pipeline dependencies.

    Each operation is ``("F", d, j, s)``, ``("B", d, j, s, zb)``,
    ``("W", d, j, s)`` or ``("FB", f_op, b_op)`` (an overlapped pair).
    Returns makespan, busy time, bubble share and the per-device timeline.
    """
    done: dict[Key, float] = {}
    free = [0.0] * p
    pos = [0] * p
    timeline: list[list[tuple]] = [[] for _ in range(p)]
    busy = [0.0] * p

    def deps(op) -> Optional[list[Key]]:
        kind = op[0]
        if kind == "FB":
            a, b = deps(op[1]), deps(op[2])
            return None if a is None or b is None else a + b
        _, d, j, s = op[:4]
        if kind == "F":
            need = [("F", d, j, s - 1)] if s > 0 else []
        elif kind == "B":
            need = [("B", d, j, s + 1)] if s < p - 1 else [("F", d, j, s)]
        else:
            need = [("B", d, j, s)]
        return need if all(k in done for k in need) else None

    def duration(op) -> tuple[float, float]:
        """(total time, compute time)."""
        kind = op[0]
        if kind == "F":
            return F + comm, F
        if kind == "B":
            b = B - W if op[4] else B
            return b + comm, b
        if kind == "W":
            return W, W
        f, b = duration(op[1])[1], duration(op[2])[1]
        return max(f + b, 2 * comm), f + b          # comm hidden behind compute

    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for dev in range(p):
            if pos[dev] >= len(orders[dev]):
                continue
            op = orders[dev][pos[dev]]
            need = deps(op)
            if need is None:
                continue
            start = max([free[dev]] + [done[k] for k in need])
            total, compute = duration(op)
            end = start + total
            for sub in (op[1:] if op[0] == "FB" else (op,)):
                done[(sub[0],) + tuple(sub[1:4])] = end
            free[dev] = end
            busy[dev] += compute
            timeline[dev].append((start, end, op))
            pos[dev] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("deadlock in pipeline schedule")
    makespan = max(free)
    return {"makespan": makespan, "busy": sum(busy) / p,
            "bubble": 1 - sum(busy) / (p * makespan), "timeline": timeline}


# ---------------------------------------------------------------- schedules
def schedule_1f1b(p: int, m: int) -> list[list[tuple]]:
    """Per-device op order of 1F1B with m micro-batches."""
    orders = []
    for s in range(p):
        warm = min(p - s - 1, m)
        ops = [("F", 0, i, s) for i in range(warm)]
        f, b = warm, 0
        while f < m:
            ops.append(("F", 0, f, s)); f += 1
            ops.append(("B", 0, b, s, False)); b += 1
        while b < m:
            ops.append(("B", 0, b, s, False)); b += 1
        orders.append(ops)
    return orders


def schedule_dualpipe(p: int, m: int) -> list[list[tuple]]:
    """Per-device op order of DualPipe (p even, m micro-batches in total,
    m/2 fed from each end, m >= 2p). Follows the eight steps of the
    reference schedule: nF0, nF0F1, nB1W1F1, nF0B1F1B0, nB1F1B0, nB1B0,
    nWB0, nW. Phase 0 is the direction in which the device is an early
    stage, phase 1 - the direction in which it is a late stage."""
    if p % 2 or m % 2 or m < 2 * p:
        raise ValueError("DualPipe needs even p, even m and m >= 2p")
    half = p // 2
    orders = []
    for r in range(p):
        h = min(r, p - 1 - r)
        dirs = (0, 1) if r < half else (1, 0)              # direction of phase 0/1
        stage = {0: h, 1: p - 1 - h}
        nf, nb = [0, 0], [0, 0]
        wq: deque = deque()
        ops: list[tuple] = []

        def F(ph):
            op = ("F", dirs[ph], nf[ph], stage[ph]); nf[ph] += 1
            return op

        def Bk(ph, zb=False):
            op = ("B", dirs[ph], nb[ph], stage[ph], zb); nb[ph] += 1
            if zb:
                wq.append(("W", dirs[ph], op[2], stage[ph]))
            return op

        def Wk():
            return wq.popleft()

        for _ in range((half - h - 1) * 2):                # 1: nF0
            ops.append(F(0))
        for _ in range(h + 1):                             # 2: nF0F1
            ops.append(F(0)); ops.append(F(1))
        for _ in range(half - h - 1):                      # 3: nB1W1F1
            ops.append(Bk(1, True)); ops.append(Wk()); ops.append(F(1))
        for i in range(m // 2 - p + h + 1):                # 4: nF0B1F1B0
            if i == 0 and h == half - 1:                   # middle rank: no overlap
                ops.append(F(0)); ops.append(Bk(1))
            else:
                ops.append(("FB", F(0), Bk(1)))
            ops.append(("FB", F(1), Bk(0)))
        for _ in range(half - h - 1):                      # 5: nB1F1B0
            ops.append(Bk(1)); ops.append(("FB", F(1), Bk(0)))
        zb = False
        for i in range(h + 1):                             # 6: nB1B0
            if i == (h + 1) // 2 and h % 2 == 1:
                zb = True
            ops.append(Bk(1, zb))
            if i == (h + 1) // 2 and h % 2 == 0:
                zb = True
            ops.append(Bk(0, zb))
        for _ in range(half - h - 1):                      # 7: nWB0
            ops.append(Wk()); ops.append(Bk(0, True))
        while wq:                                          # 8: nW
            ops.append(Wk())
        orders.append(ops)
    return orders


def compare_schedules(p: int, m: int, F: float, B: float, W: float, comm: float = 0.0) -> dict:
    """Simulate 1F1B (m micro-batches) and DualPipe (m in total) side by side."""
    res = {}
    ideal = m * (F + B)
    for name, orders in (("1F1B", schedule_1f1b(p, m)), ("DualPipe", schedule_dualpipe(p, m))):
        r = simulate(orders, p, F, B, W, comm)
        res[name] = {"makespan": r["makespan"], "bubble_share": r["bubble"],
                     "bubble_time": r["makespan"] - ideal, "timeline": r["timeline"]}
    return res


def ascii_timeline(timeline: list[list[tuple]], makespan: float, unit: float = 1.0) -> str:
    """One row per device; F = forward, B = backward, W = weight grad,
    X = overlapped F+B pair, '.' = bubble (lower-case: second direction)."""
    rows = []
    width = int(round(makespan / unit))
    for dev, ops in enumerate(timeline):
        line = ["."] * width
        for st, en, op in ops:
            kind = op[0]
            ch = "X" if kind == "FB" else kind
            if kind != "FB" and op[1] == 1:
                ch = ch.lower()
            for t in range(int(round(st / unit)), int(round(en / unit))):
                if t < width:
                    line[t] = ch
        rows.append(f"dev {dev}: " + "".join(line))
    return "\n".join(rows)


# ------------------------------------------------------- measured stage times
def measure_stage_times(config=None, p: int = 4, micro_batch: int = 4, seq_len: int = 256,
                        repeats: int = 3) -> dict:
    """Measure F, B and W of one TinySeek block on this CPU and scale them to
    one of ``p`` equal pipeline stages ("virtual accelerators").

    W is estimated as B(full) - B(input gradient only)."""
    from .config import get_preset
    from .model import Block
    cfg = config or get_preset("tiny_v3")
    torch.manual_seed(0)
    blk = Block(cfg, cfg.num_hidden_layers - 1)
    x = torch.randn(micro_batch, seq_len, cfg.hidden_size)
    pos = torch.arange(seq_len)

    def timed(fn) -> float:
        fn()
        t = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t) / repeats

    def fwd():
        with torch.no_grad():
            blk(x, pos)

    def full():
        xi = x.clone().requires_grad_(True)
        blk(xi, pos).sum().backward()
        blk.zero_grad(set_to_none=True)

    def inp_only():
        for q in blk.parameters():
            q.requires_grad_(False)
        xi = x.clone().requires_grad_(True)
        blk(xi, pos).sum().backward()
        for q in blk.parameters():
            q.requires_grad_(True)

    f = timed(fwd)
    fb = timed(full)
    fi = timed(inp_only)
    B, Bi = max(fb - f, 1e-9), max(fi - f, 1e-9)
    W = max(B - Bi, 0.0)
    layers_per_stage = cfg.num_hidden_layers / p
    return {"F": f * layers_per_stage, "B": B * layers_per_stage,
            "W": W * layers_per_stage, "layers_per_stage": layers_per_stage}


# -------------------------------------------------------------- all-to-all
def all_to_all_bytes(topk: torch.Tensor, n_experts: int, ep: int, hidden: int,
                     dispatch_bytes: float = 1.0, combine_bytes: float = 2.0,
                     scale_block: int = 128) -> dict:
    """Traffic of one MoE layer for routing decisions ``topk`` (tokens, K).

    Experts are split contiguously over ``ep`` devices and tokens evenly
    (token t lives on device t * ep // n). Dispatch sends FP8 activations
    plus one FP32 scale per ``scale_block`` channels; combine returns BF16.
    ``naive`` sends a copy per remote expert, ``dedup`` one copy per remote
    device (V3 sends each token to a node once and forwards it inside).
    """
    n, K = topk.shape
    per_dev = n_experts // ep
    dev_of_expert = topk // per_dev
    home = (torch.arange(n) * ep // n)[:, None]
    remote = dev_of_expert != home
    naive = int(remote.sum())
    dedup = 0
    for t in range(n):
        dedup += len(set(dev_of_expert[t][remote[t]].tolist()))
    per_tok_dispatch = hidden * dispatch_bytes + 4 * hidden / scale_block
    per_tok_combine = hidden * combine_bytes
    return {"copies_naive": naive, "copies_dedup": dedup,
            "bytes_naive": naive * (per_tok_dispatch + per_tok_combine),
            "bytes_dedup": dedup * (per_tok_dispatch + per_tok_combine),
            "bytes_per_token_dedup": dedup * (per_tok_dispatch + per_tok_combine) / n}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Simulate 1F1B and DualPipe for TinySeek "
                                "on virtual accelerators (Chapter 11).")
    p.add_argument("--p", type=int, default=4, help="pipeline stages (even)")
    p.add_argument("--m", type=int, default=16, help="micro-batches")
    p.add_argument("--comm", type=float, default=0.0,
                   help="all-to-all time per chunk, in units of F")
    p.add_argument("--measure", action="store_true", help="measure F/B/W on this CPU")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.measure:
        t = measure_stage_times(p=args.p)
        F, B, W = 1.0, t["B"] / t["F"], t["W"] / t["F"]
        print(f"measured per stage: F={t['F'] * 1e3:.1f} ms, B={t['B'] * 1e3:.1f} ms, "
              f"W={t['W'] * 1e3:.1f} ms -> in units of F: B={B:.2f}, W={W:.2f}")
    else:
        F, B, W = 1.0, 2.0, 1.0
    res = compare_schedules(args.p, args.m, F, B, W, args.comm * F)
    for name, r in res.items():
        print(f"{name:9s} makespan {r['makespan']:7.2f}  bubble {r['bubble_time']:6.2f} "
              f"({r['bubble_share']:.1%})")
    print(f"formulas: 1F1B {bubble_formula('1F1B', args.p, F, B):.2f}, "
          f"DualPipe {bubble_formula('DualPipe', args.p, F, B, W):.2f}")
    if args.p * args.m <= 256 and not args.measure:
        print(ascii_timeline(res["DualPipe"]["timeline"], res["DualPipe"]["makespan"]))


if __name__ == "__main__":
    main()
