"""Lab 6. SFT and GRPO on a verifiable task (Chapters 12-13).

Steps:
  1. evaluate the pre-trained model on arithmetic (greedy accuracy, format of
     the answer, mean rule-based reward of sampled answers);
  2. SFT with prompt masking (formula (12.1)) on arithmetic instructions in the
     ``<think>...</think><answer>...</answer>`` format;
  3. GRPO (formulas (13.1)-(13.5)) from the SFT checkpoint with the rule-based
     reward r = r_format + r_acc (13.6): no critic, group-relative advantages;
  4. compare base -> SFT -> GRPO and plot the reward and the answer length.

Checkpoints: runs/lab6/sft and runs/lab6/grpo (used by labs 7-8).

Run (from code/):
    python tinyseek_labs/lab6_rl.py
    python tinyseek_labs/lab6_rl.py --group-size 4 --beta 0.0 --grpo-steps 60
Outputs: runs/lab6/summary.csv, grpo_log.csv, reward.svg
"""
import argparse
from _common import Timer, base_model, print_table, run_dir, smooth, svg_plot, write_csv

from tinyseek.evaluate import arith_accuracy
from tinyseek.grpo import grpo_train
from tinyseek.sft import arith_sft_examples, sft
from tinyseek.train import load_checkpoint, save_checkpoint


def evaluate(model, tok, digits, n_eval):
    g = arith_accuracy(model, tok, n=n_eval, digits=digits, seed=2024)
    s = arith_accuracy(model, tok, n=n_eval // 2, digits=digits, seed=2025, temperature=1.0,
                       samples=4)
    return {"greedy_acc": round(g["accuracy"], 3), "format": round(g["format"], 3),
            "sampled_reward(tau=1)": round(s["reward"], 3),
            "sampled_acc(tau=1)": round(s["accuracy"], 3), "answer_len": round(g["length"], 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sft-steps", type=int, default=400)
    ap.add_argument("--sft-lr", type=float, default=1e-3)
    ap.add_argument("--grpo-steps", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=8, help="G")
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="GRPO learning rate")
    ap.add_argument("--beta", type=float, default=0.04, help="KL weight")
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--digits", default="1,2", help="operand lengths")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(6), Timer()
    digits = [int(d) for d in args.digits.split(",")]
    model, tok, base_dir = base_model()
    rows = [{"stage": "base", **evaluate(model, tok, digits, args.n_eval)}]

    # SFT -------------------------------------------------------------------------
    ex = arith_sft_examples(tok, 20000, digits, seed=args.seed + 1)
    hist = sft(model, ex, tok.pad_id, args.sft_steps, lr=args.sft_lr, seed=args.seed,
               verbose=False)
    save_checkpoint(model, out / "sft", base_dir / "tokenizer.json")
    rows.append({"stage": "SFT", **evaluate(model, tok, digits, args.n_eval)})
    print(f"SFT: loss {hist[0]['loss']:.3f} -> {hist[-1]['loss']:.3f} "
          f"in {hist[-1]['time']:.0f} s")

    # GRPO ------------------------------------------------------------------------
    policy, _ = load_checkpoint(out / "sft")           # a fresh copy of the SFT model
    log = grpo_train(policy, tok, args.grpo_steps, args.prompts_per_step, args.group_size,
                     args.lr, args.beta, args.eps_clip, digits=digits, seed=args.seed,
                     log_path=out / "grpo_log.jsonl", verbose=True)
    save_checkpoint(policy, out / "grpo", base_dir / "tokenizer.json")
    rows.append({"stage": "SFT+GRPO", **evaluate(policy, tok, digits, args.n_eval)})
    print_table(rows, "Arithmetic: base -> SFT -> GRPO "
                      f"(greedy on {args.n_eval} problems; tau=1 with 4 samples each)")
    write_csv(out / "summary.csv", rows)
    write_csv(out / "grpo_log.csv", log)
    k = max(1, args.grpo_steps // 2)
    first = sum(r["reward"] for r in log[:k]) / k
    last = sum(r["reward"] for r in log[-k:]) / k
    print(f"mean training reward: first half {first:.3f} -> second half {last:.3f}")
    steps = [r["step"] for r in log]
    svg_plot(out / "reward.svg",
             {"reward": (steps, smooth([r["reward"] for r in log], 5)),
              "accuracy": (steps, smooth([r["accuracy"] for r in log], 5)),
              "length/40": (steps, [r["length"] / 40 for r in log])},
             "GRPO on arithmetic (moving average)", "step", "value")
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
