"""Lab 7. Distilling reasoning into a smaller model (Chapter 14).

Steps:
  1. take the TinySeek-V3 teacher after SFT/GRPO (lab 6) and measure it;
  2. rejection sampling: the teacher writes several solutions per problem,
     only verified ones are kept (Section 14.4);
  3. train three identical small dense students for the same number of steps:
     on the teacher's traces (sequence-level distillation, (14.4)), with the
     token-level KD loss (14.2) at temperature tau, and on answers without a
     chain of thought;
  4. compare accuracy, size and speed of the students with the teacher.

Run (from code/):
    python tinyseek_labs/lab7_distill.py
    python tinyseek_labs/lab7_distill.py --tau 4 --alpha 0.9 --student-width 96
Outputs: runs/lab7/students.csv, loss.svg
"""
import argparse
import time

from _common import RUNS, Timer, print_table, run_dir, sft_model, smooth, svg_plot, write_csv
import torch

from tinyseek.distill import collect_teacher_traces, distill, student_config
from tinyseek.evaluate import arith_accuracy
from tinyseek.model import count_parameters
from tinyseek.train import load_checkpoint


@torch.no_grad()
def tokens_per_second(model, n: int = 64) -> float:
    ids = torch.randint(0, 2048, (1, 16))
    cache = model.new_cache()
    model(ids, cache=cache)
    t0 = time.perf_counter()
    for _ in range(n):
        model(ids[:, :1], cache=cache)
    return n / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=600, help="student training steps")
    ap.add_argument("--n-prompts", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=4, help="teacher samples per prompt")
    ap.add_argument("--tau", type=float, default=2.0, help="KD temperature")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the KL term")
    ap.add_argument("--student-width", type=int, default=64)
    ap.add_argument("--student-layers", type=int, default=2)
    ap.add_argument("--digits", default="1,2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(7), Timer()
    digits = [int(d) for d in args.digits.split(",")]
    if (RUNS / "lab6" / "grpo" / "model.pt").exists():
        teacher, tok = load_checkpoint(RUNS / "lab6" / "grpo")
        print("teacher: runs/lab6/grpo")
    else:
        teacher, tok, _ = sft_model()
    t_acc = arith_accuracy(teacher, tok, n=300, digits=digits, seed=77)
    rows = [{"model": "teacher tiny_v3", "params": count_parameters(teacher),
             "active": count_parameters(teacher, active=True),
             "accuracy": round(t_acc["accuracy"], 3), "format": round(t_acc["format"], 3),
             "tok_per_s": round(tokens_per_second(teacher), 1)}]

    col = collect_teacher_traces(teacher, tok, args.n_prompts, args.samples, 0.7, digits,
                                 args.seed)
    print(f"rejection sampling: {col['sample_accuracy']:.1%} of teacher samples correct, "
          f"{len(col['traces'])} of {args.n_prompts} prompts kept ({col['kept_prompts']:.1%})")
    print(f"example trace: {col['traces'][0][0]!r} -> {col['traces'][0][1]!r}")

    scfg = student_config(teacher.config.vocab_size).replace(
        hidden_size=args.student_width, num_hidden_layers=args.student_layers,
        num_attention_heads=max(1, args.student_width // 32),
        intermediate_size=3 * args.student_width, first_k_dense_replace=args.student_layers)
    curves = {}
    for method in ("traces", "kd", "answer_only"):
        student, info = distill(teacher, tok, method, args.steps, args.n_prompts, tau=args.tau,
                                alpha=args.alpha, digits=digits, student_cfg=scfg,
                                traces=col["traces"], seed=args.seed, verbose=False)
        acc = arith_accuracy(student, tok, n=300, digits=digits, seed=77)
        rows.append({"model": f"student: {method}", "params": count_parameters(student),
                     "active": count_parameters(student, active=True),
                     "accuracy": round(acc["accuracy"], 3), "format": round(acc["format"], 3),
                     "tok_per_s": round(tokens_per_second(student), 1)})
        h = info["history"]
        curves[method] = ([r["step"] for r in h], smooth([r["loss"] for r in h], 5))
    print_table(rows, f"Teacher vs students ({args.steps} steps each, "
                      f"tau={args.tau}, alpha={args.alpha}); accuracy on 300 problems")
    write_csv(out / "students.csv", rows)
    svg_plot(out / "loss.svg", curves, "Student training loss", "step", "loss")
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
