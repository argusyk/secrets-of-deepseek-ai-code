"""Shared helpers for the TinySeek laboratory scripts.

* `run_dir(n)` - ``runs/labN`` for the outputs of lab N;
* `print_table` / `write_csv` - the results table on screen and as CSV;
* `svg_plot` - a dependency-free SVG line chart (no matplotlib needed);
* `base_model` / `sft_model` - reuse checkpoints of earlier labs or build
  small ones on the fly, so every lab can also run on its own.
"""
from __future__ import annotations

import csv
import os
import html
import math
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# idle OpenMP threads sleep instead of spinning: several TinySeek processes
# sharing a laptop CPU then slow each other down far less
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

RUNS = Path("runs")
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]


def run_dir(n: int) -> Path:
    d = RUNS / f"lab{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def __str__(self) -> str:
        return f"{time.time() - self.t0:.1f} s"


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def print_table(rows: Sequence[dict], title: Optional[str] = None) -> None:
    """Print a list of dicts as an aligned text table."""
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    cells = [[_fmt(r.get(c, "")) for c in cols] for r in rows]
    w = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    if title:
        print(f"\n{title}")
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    print("  ".join("-" * w[i] for i in range(len(cols))))
    for row in cells:
        print("  ".join(row[i].ljust(w[i]) for i in range(len(cols))))


def write_csv(path: Path, rows: Sequence[dict]) -> Path:
    if not rows:
        return path
    cols = list(rows[0].keys())
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return path


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9 * step:
        out.append(round(v, 10))
        v += step
    return out


def svg_plot(path: Path, series: dict[str, tuple[Sequence[float], Sequence[float]]],
             title: str, xlabel: str, ylabel: str, logy: bool = False,
             width: int = 720, height: int = 420) -> Path:
    """Write a simple multi-series line chart as SVG."""
    ml, mr, mt, mb = 70, 170, 40, 55
    pw, ph = width - ml - mr, height - mt - mb
    xs = [x for xv, _ in series.values() for x in xv]
    ys = [y for _, yv in series.values() for y in yv if y is not None and not math.isnan(y)]
    if logy:
        ys = [math.log10(max(y, 1e-12)) for y in ys]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or 0.5
    y0, y1 = y0 - pad, y1 + pad
    x1 = x1 if x1 > x0 else x0 + 1

    def sx(x):
        return ml + (x - x0) / (x1 - x0) * pw

    def sy(y):
        y = math.log10(max(y, 1e-12)) if logy else y
        return mt + (1 - (y - y0) / (y1 - y0)) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{ml}" y="22" font-size="14" font-weight="bold">{html.escape(title)}</text>',
           f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>']
    for t in _ticks(x0, x1):
        X = sx(t)
        out.append(f'<line x1="{X:.1f}" y1="{mt}" x2="{X:.1f}" y2="{mt + ph}" stroke="#eee"/>')
        out.append(f'<text x="{X:.1f}" y="{mt + ph + 16}" text-anchor="middle">{t:g}</text>')
    for t in _ticks(y0, y1):
        Y = mt + (1 - (t - y0) / (y1 - y0)) * ph
        lab = f"{10 ** t:.3g}" if logy else f"{t:g}"
        out.append(f'<line x1="{ml}" y1="{Y:.1f}" x2="{ml + pw}" y2="{Y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{ml - 6}" y="{Y + 4:.1f}" text-anchor="end">{lab}</text>')
    out.append(f'<text x="{ml + pw / 2}" y="{height - 12}" text-anchor="middle">'
               f'{html.escape(xlabel)}</text>')
    out.append(f'<text x="16" y="{mt + ph / 2}" text-anchor="middle" '
               f'transform="rotate(-90 16 {mt + ph / 2})">{html.escape(ylabel)}</text>')
    for i, (name, (xv, yv)) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xv, yv)
                       if y is not None and not math.isnan(y))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        ly = mt + 14 + 18 * i
        out.append(f'<line x1="{ml + pw + 12}" y1="{ly - 4}" x2="{ml + pw + 32}" '
                   f'y2="{ly - 4}" stroke="{col}" stroke-width="3"/>')
        out.append(f'<text x="{ml + pw + 38}" y="{ly}">{html.escape(name)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def smooth(values: Sequence[float], k: int = 5) -> list[float]:
    """Moving average for noisy training curves."""
    out = []
    for i in range(len(values)):
        w = values[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def base_model(steps: int = 600, seed: int = 0, verbose: bool = True):
    """The pre-trained tiny_v3 of lab 4 (runs/lab4/v3) or, if absent, a quick
    one trained here (``steps`` steps) and saved to runs/base_v3."""
    from tinyseek.config import get_preset
    from tinyseek.data import PackedDataset, prepare_data
    from tinyseek.train import load_checkpoint, save_checkpoint, train
    for d in (RUNS / "lab4" / "v3", RUNS / "base_v3"):
        if (d / "model.pt").exists():
            if verbose:
                print(f"using pre-trained checkpoint {d}")
            return load_checkpoint(d) + (d,)
    cfg = get_preset("tiny_v3")
    tok, data = prepare_data(RUNS / "data", cfg.vocab_size, seed)
    if verbose:
        print(f"no lab 4 checkpoint: pre-training tiny_v3 for {steps} steps ...")
    model, _ = train(cfg, PackedDataset(data["train"], 256), steps, seed=seed, verbose=verbose)
    d = save_checkpoint(model, RUNS / "base_v3", RUNS / "data" / f"tokenizer_{cfg.vocab_size}.json")
    return model, tok, d


def sft_model(steps: int = 400, seed: int = 0, verbose: bool = True):
    """The SFT model of lab 6 (runs/lab6/sft) or a quick one built here from
    `base_model` and saved to runs/base_sft."""
    from tinyseek.sft import arith_sft_examples, sft
    from tinyseek.train import load_checkpoint, save_checkpoint
    for d in (RUNS / "lab6" / "sft", RUNS / "base_sft"):
        if (d / "model.pt").exists():
            if verbose:
                print(f"using SFT checkpoint {d}")
            return load_checkpoint(d) + (d,)
    model, tok, base_dir = base_model(seed=seed, verbose=verbose)
    if verbose:
        print(f"no lab 6 checkpoint: SFT for {steps} steps ...")
    sft(model, arith_sft_examples(tok, 20000, (1, 2), seed=seed + 1), tok.pad_id, steps, seed=seed,
        verbose=verbose)
    d = save_checkpoint(model, RUNS / "base_sft", base_dir / "tokenizer.json")
    return model, tok, d
