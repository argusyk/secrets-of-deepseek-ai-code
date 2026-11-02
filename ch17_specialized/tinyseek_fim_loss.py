"""Does FIM in the corpus teach TinySeek to use the suffix? (Section 17.9)

For each FIM task of tinyseek.evaluate (one identifier hidden in held-out
standard-library code) the script measures how well a checkpoint predicts
the hidden middle, in bits per character, under two prompts:

    PSM          <|fim_begin|>prefix<|fim_hole|>suffix<|fim_end|>  -> middle
    prefix only  prefix                                          -> middle

A model that learned FIM should predict the middle better when it also sees
the suffix (gain > 0). A model trained without FIM has never seen the FIM
tokens and gains nothing, or loses. Bits per character, unlike per-token
loss, stay comparable between checkpoints with different tokenizers.

Run from code/ (checkpoints from the TinySeek step of Chapter 17):
    python ch17_specialized/tinyseek_fim_loss.py runs/v3 runs/v3_nofim
"""
import argparse
import math
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def middle_bits(model, tok, context, middle):
    """Bits per character of `middle` given `context` (teacher forcing)."""
    import torch
    ctx, mid = tok.encode(context), tok.encode(middle)
    room = model.config.max_position_embeddings - len(mid)
    ids = torch.tensor([(ctx[-room:] if room > 0 else []) + mid])
    with torch.no_grad():
        logits, _ = model(ids)
    logp = torch.log_softmax(logits[0, :-1].float(), -1)
    tgt = ids[0, 1:]
    nll = -logp[torch.arange(len(tgt)), tgt][-len(mid):].sum().item()
    return nll / math.log(2) / max(1, len(middle))


def fim_gain(run_dir, data_dir, n=40, seed=0):
    from tinyseek.evaluate import fim_prompt, make_fim_tasks
    from tinyseek.train import load_checkpoint
    model, tok = load_checkpoint(run_dir)
    tasks = make_fim_tasks(data_dir, n=n, seed=seed)
    psm = [middle_bits(model, tok, fim_prompt(t), t["middle"])
           for t in tasks]
    plain = [middle_bits(model, tok, t["prefix"], t["middle"])
             for t in tasks]
    psm_m, plain_m = sum(psm) / len(psm), sum(plain) / len(plain)
    better = sum(a < b for a, b in zip(psm, plain)) / len(tasks)
    return psm_m, plain_m, better


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", default=["runs/v3"])
    ap.add_argument("--data", default="runs/data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = [r for r in a.runs if (Path(r) / "model.pt").exists()]
    if not runs or not (Path(a.data) / "corpus.jsonl").exists():
        print("No TinySeek checkpoint or corpus found. From code/ run:\n"
              "  python -m tinyseek.train --preset tiny_v3 --steps 1500 "
              "--out runs/v3\n  python ch17_specialized/"
              "tinyseek_fim_loss.py runs/v3")
        return
    print(f"{'checkpoint':28s} {'PSM':>7s} {'prefix':>7s} {'gain':>7s}"
          f"  PSM better on")
    for r in runs:
        psm, plain, better = fim_gain(r, a.data, a.n, a.seed)
        print(f"{r[-28:]:28s} {psm:7.2f} {plain:7.2f} {plain - psm:7.2f}"
              f"  {better:.0%} of tasks   (bits per character)")


if __name__ == "__main__":
    main()
