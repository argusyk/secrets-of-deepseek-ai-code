"""Lab 1. Corpus and tokenizer: BPE, fertility, FIM, deduplication (Chapters 2-3).

Steps:
  1. build the offline corpus (stdlib code + UK/EN prose + arithmetic) with
     MinHash deduplication of injected near-copies;
  2. train byte-level BPE tokenizers of several sizes and measure fertility
     (tokens per word) and bytes per token on held-out UK, EN and code;
  3. show segmentations and a FIM (PSM) sample;
  4. measure how MinHash/LSH detection depends on how much a copy was edited.

Run (from code/):
    python tinyseek_labs/lab1_tokenizer.py
    python tinyseek_labs/lab1_tokenizer.py --vocab-sizes 384,768,1536 --bands 8
Outputs: runs/lab1/fertility.csv, dedup.csv, fertility.svg
"""
import argparse
import random
import time

import numpy as np
from _common import Timer, print_table, run_dir, svg_plot, write_csv

from tinyseek.data import (_perturb, build_corpus, load_corpus, minhash_signatures,
                           minhash_dedup)
from tinyseek.tokenizer import ByteBPETokenizer, fim_inverse, fim_transform


def jaccard(a: str, b: str, k: int) -> float:
    def sh(t):
        t = " ".join(t.lower().split())
        return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}
    x, y = sh(a), sh(b)
    return len(x & y) / len(x | y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vocab-sizes", default="512,1024,2048,4096")
    ap.add_argument("--threshold", type=float, default=0.8, help="MinHash threshold")
    ap.add_argument("--num-perm", type=int, default=64)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--shingle", type=int, default=5, help="k of character k-grams")
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(1), Timer()

    # 1. corpus -----------------------------------------------------------
    stats = build_corpus(out / "data", fim_rate=args.fim_rate, seed=args.seed,
                         dedup_threshold=args.threshold, verbose=False)
    print_table([{"source": s, "KB": round(b / 1000, 1)} for s, b in stats["bytes"].items()],
                "Corpus composition")
    print(f"near-duplicates removed: {stats['near_duplicates_removed']} "
          f"(injected {stats['injected_duplicates']}, caught {stats['injected_found']})")
    corpus = load_corpus(out / "data")
    train_texts = [r["text"] for r in corpus if r["split"] == "train"]
    val = {s: "\n".join(r["text"] for r in corpus if r["split"] == "val" and r["source"] == s)
           for s in ("uk", "en", "code")}
    fim_share = np.mean([r["text"].startswith("<|fim_begin|>") for r in corpus
                         if r["source"] == "code" and r["split"] == "train"])
    print(f"FIM share among training code documents: {fim_share:.2f}")

    # 2. tokenizers of several sizes ---------------------------------------
    rows, toks = [], {}
    for V in [int(v) for v in args.vocab_sizes.split(",")]:
        t0 = time.time()
        tok = ByteBPETokenizer().train(train_texts, V)
        toks[V] = tok
        row = {"vocab": tok.vocab_size, "train_s": round(time.time() - t0, 1)}
        for s, text in val.items():
            row[f"fert_{s}"] = round(tok.fertility(text), 3)
            row[f"bpt_{s}"] = round(tok.bytes_per_token(text), 3)
        rows.append(row)
    print_table(rows, "Fertility (tokens/word) and bytes per token on held-out text")
    write_csv(out / "fertility.csv", rows)
    svg_plot(out / "fertility.svg",
             {s: ([r["vocab"] for r in rows], [r[f"fert_{s}"] for r in rows])
              for s in ("uk", "en", "code")},
             "Fertility vs vocabulary size", "vocabulary size", "tokens per word")

    # 3. segmentations and FIM ----------------------------------------------
    tok = toks[max(toks)]
    print(f"\nSegmentation with V={max(toks)}:")
    for w in ["латентна увага", "latent attention", "def forward(self, x):", "47+38=85"]:
        print(f"  {w!r:26} -> {' | '.join(tok.token_str(i) for i in tok.encode(w))}")
    rng = random.Random(args.seed)
    sample = "def area(r):\n    return 3.14159 * r * r\n"
    fim = fim_transform(sample, 1.0, rng)
    print(f"\nFIM (PSM) sample:\n  {fim!r}\n  inverse restores the text: {fim_inverse(fim) == sample}")

    # 4. dedup sensitivity --------------------------------------------------
    base = [r["text"] for r in corpus if r["source"] in ("uk", "en")][:60]
    drng = random.Random(args.seed)
    drows = []
    for rate in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        copies = [_perturb(d, drng, rate) for d in base]
        docs = base + copies
        keep, _ = minhash_dedup(docs, args.threshold, args.num_perm, args.bands, args.shingle,
                                args.seed)
        caught = sum(1 for i in range(len(base), len(docs)) if i not in set(keep))
        sig = minhash_signatures([base[0], copies[0]], args.num_perm, args.shingle, args.seed)
        drows.append({"edit_rate": rate,
                      "jaccard": round(np.mean([jaccard(a, b, args.shingle)
                                                for a, b in zip(base, copies)]), 3),
                      "minhash_est_doc0": round(float(np.mean(sig[0] == sig[1])), 3),
                      "recall": round(caught / len(base), 3),
                      "false_merges": len(base) - len([i for i in keep if i < len(base)])})
    print_table(drows, f"MinHash/LSH: threshold {args.threshold}, {args.num_perm} permutations, "
                       f"{args.bands} bands, k={args.shingle}")
    write_csv(out / "dedup.csv", drows)
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
