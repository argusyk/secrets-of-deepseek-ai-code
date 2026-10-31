"""Direct Preference Optimization (DPO) loss on toy log-probabilities.

Teaching implementation in NumPy for Chapter 12 of "Secrets of DeepSeek AI".
For a preference pair (chosen y_w, rejected y_l) DPO compares how much the
policy has moved away from a frozen reference model on each answer:

    margin = beta * [(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))]
    loss   = -log sigmoid(margin)

The quantity beta * (log pi - log ref) acts as an implicit reward.

Run:
    python dpo_loss.py
"""
import numpy as np


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """All arguments are sequence log-probabilities (sums over tokens)."""
    r_w = beta * (pi_w - ref_w)          # implicit reward of the chosen answer
    r_l = beta * (pi_l - ref_l)          # implicit reward of the rejected answer
    margin = r_w - r_l
    return -log_sigmoid(margin), r_w, r_l


def grad_wrt_logps(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """d loss / d (log pi(y_w)) and d loss / d (log pi(y_l)).
    The weight sigma(-margin) is large when the model still ranks wrongly."""
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    w = 1.0 / (1.0 + np.exp(margin))      # sigma(-margin)
    return -beta * w, beta * w


if __name__ == "__main__":
    ref_w, ref_l = -42.0, -40.0          # the reference slightly prefers y_l
    print("policy starts equal to the reference:")
    loss, r_w, r_l = dpo_loss(ref_w, ref_l, ref_w, ref_l)
    print(f"  loss = {loss:.4f} (= ln 2 = {np.log(2):.4f}), rewards {r_w:.2f} / {r_l:.2f}")

    print("simple gradient steps on the two sequence log-probs:")
    pi_w, pi_l = ref_w, ref_l
    for step in range(6):
        gw, gl = grad_wrt_logps(pi_w, pi_l, ref_w, ref_l)
        pi_w -= 20.0 * gw                 # raise log pi(y_w)
        pi_l -= 20.0 * gl                 # lower log pi(y_l)
        loss, r_w, r_l = dpo_loss(pi_w, pi_l, ref_w, ref_l)
        print(f"  step {step + 1}: loss {loss:.4f}  implicit rewards "
              f"chosen {r_w:+.3f} rejected {r_l:+.3f}")

    print("effect of beta on the loss for the same log-prob shift (+2 / -2):")
    for beta in (0.05, 0.1, 0.5, 1.0):
        loss, _, _ = dpo_loss(ref_w + 2, ref_l - 2, ref_w, ref_l, beta)
        print(f"  beta {beta:<4}: loss {loss:.4f}")
