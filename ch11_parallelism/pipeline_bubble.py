"""Pipeline-parallel bubbles: closed-form formulas and a tiny 1F1B simulator.

Accompanies Chapter 11 of "Secrets of DeepSeek AI". Pure Python, no deps.

* gpipe_bubble_fraction(p, m)      -- idle share of a GPipe schedule
* bubble_time(kind, p, F, B, W, FB) -- bubble formulas compared in the
                                      DeepSeek-V3 report (1F1B, ZB1P, DualPipe)
* simulate_1f1b(p, m, F, B)        -- event simulation of a 1F1B schedule:
                                      measures the bubble and prints an ASCII chart

Run:
    python pipeline_bubble.py
"""


def gpipe_bubble_fraction(p, m):
    """Share of idle time for p stages and m micro-batches: (p-1)/(m+p-1)."""
    return (p - 1) / (m + p - 1)


def bubble_time(kind, p, F, B, W=0.0, FB=None):
    """Bubble length per the DeepSeek-V3 report.

    F  - forward chunk time, B - full backward chunk time,
    W  - "backward for weights" part of B, FB - time of a mutually
    overlapped forward+backward chunk pair (DualPipe only).
    """
    if kind == "1F1B":
        return (p - 1) * (F + B)
    if kind == "ZB1P":
        return (p - 1) * (F + B - 2 * W)
    if kind == "DualPipe":
        if p % 2:
            raise ValueError("DualPipe needs an even number of stages")
        FB = F + B if FB is None else FB
        return (p / 2 - 1) * (FB + B - 3 * W)
    raise ValueError(kind)


MEMORY = {  # (parameter copies, in-flight activations) per the report
    "1F1B": ("1x", "p"),
    "ZB1P": ("1x", "p"),
    "DualPipe": ("2x", "p+1"),
}


def _ops_1f1b(stage, p, m):
    """Order of operations on one stage in the classic 1F1B schedule."""
    warm = min(p - stage - 1, m)
    ops = [("F", i) for i in range(warm)]
    f, b = warm, 0
    while f < m:                          # steady state: one F, one B
        ops.append(("F", f)); f += 1
        ops.append(("B", b)); b += 1
    while b < m:                          # cool-down: remaining backwards
        ops.append(("B", b)); b += 1
    return ops


def simulate_1f1b(p, m, F=1, B=2):
    """Event simulation. Returns (makespan, idle_fraction, timeline) where
    timeline[s] is a list of (start, end, kind, microbatch) for stage s."""
    done = {}                             # (kind, stage, mb) -> finish time
    free = [0] * p
    timeline = [[] for _ in range(p)]
    pending = [list(_ops_1f1b(s, p, m)) for s in range(p)]
    while any(pending):
        progressed = False
        for s in range(p):
            if not pending[s]:
                continue
            kind, mb = pending[s][0]
            if kind == "F":
                dep = done.get(("F", s - 1, mb), 0) if s > 0 else 0
                ready = s == 0 or ("F", s - 1, mb) in done
            else:
                dep = done.get(("B", s + 1, mb), 0) if s < p - 1 else done.get(("F", s, mb), 0)
                ready = (("B", s + 1, mb) in done) if s < p - 1 else (("F", s, mb) in done)
            if not ready:
                continue
            start = max(free[s], dep)
            end = start + (F if kind == "F" else B)
            free[s] = end
            done[(kind, s, mb)] = end
            timeline[s].append((start, end, kind, mb))
            pending[s].pop(0)
            progressed = True
        if not progressed:
            raise RuntimeError("deadlock in schedule")
    makespan = max(free)
    busy = sum(e - st for row in timeline for st, e, _, _ in row)
    return makespan, 1 - busy / (p * makespan), timeline


def ascii_chart(timeline, makespan):
    """Digits = forward of micro-batch i, letters = its backward, '.' = bubble."""
    rows = []
    for s, row in enumerate(timeline):
        line = ["."] * makespan
        for st, e, kind, mb in row:
            ch = str(mb + 1) if kind == "F" else chr(ord("a") + mb)
            for t in range(st, e):
                line[t] = ch
        rows.append(f"stage {s}: " + "".join(line))
    return "\n".join(rows)


if __name__ == "__main__":
    p, m = 4, 8
    span, idle, tl = simulate_1f1b(p, m, F=1, B=2)
    print(f"1F1B simulation, p={p}, m={m}, F=1, B=2")
    print(ascii_chart(tl, span))
    print(f"makespan={span}, measured bubble per stage={span - 3 * m}, "
          f"formula (p-1)(F+B)={bubble_time('1F1B', p, 1, 2)}")
    print(f"idle fraction={idle:.3f}, GPipe formula (p-1)/(m+p-1)={gpipe_bubble_fraction(p, m):.3f}")
    print()
    print("Bubble length for p=16 (illustrative F=1, B=2, W=1, FB=F+B):")
    for kind in ("1F1B", "ZB1P", "DualPipe"):
        t = bubble_time(kind, 16, F=1, B=2, W=1)
        params, act = MEMORY[kind]
        print(f"  {kind:9s} bubble={t:5.1f}  params={params:3s} activations={act}")
    print()
    print("GPipe idle share for p=16 as micro-batches grow:")
    for m in (16, 32, 64, 128):
        print(f"  m={m:4d}: {gpipe_bubble_fraction(16, m):.3f}")
