"""Group Relative Policy Optimization (GRPO) on a toy problem.

Teaching implementation for Chapter 13 of "Secrets of DeepSeek AI".
The "policy" is a softmax over K candidate answers to one question; one of
them is correct. Each step we sample a group of G answers, score them with a
rule-based reward, turn rewards into group-relative advantages, and take a
few gradient steps on the clipped GRPO objective with a KL penalty to a
frozen reference policy. No critic (value model) is involved.

Run:
    python grpo.py
"""
import numpy as np


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def group_advantages(rewards, eps=1e-6):
    """A_i = (r_i - mean(r)) / std(r), computed inside one group (formula 13.3)."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def kl_k3(p_theta, p_ref):
    """Unbiased, non-negative per-sample KL estimate used by GRPO (formula 13.2):
    pi_ref/pi_theta - log(pi_ref/pi_theta) - 1."""
    x = p_ref / p_theta
    return x - np.log(x) - 1.0


def grpo_objective(logits, logits_old, logits_ref, samples, adv, eps_clip=0.2, beta=0.04):
    """Value of the GRPO surrogate (formula 13.1) for one group of samples."""
    p, p_old, p_ref = softmax(logits), softmax(logits_old), softmax(logits_ref)
    ratio = p[samples] / p_old[samples]
    surr = np.minimum(ratio * adv, np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * adv)
    return np.mean(surr - beta * kl_k3(p[samples], p_ref[samples]))


def grpo_grad(logits, logits_old, logits_ref, samples, adv, eps_clip=0.2, beta=0.04):
    """Analytic gradient of grpo_objective w.r.t. the logits (softmax policy)."""
    K = logits.shape[0]
    p, p_old, p_ref = softmax(logits), softmax(logits_old), softmax(logits_ref)
    g = np.zeros(K)
    for o, a in zip(samples, adv):
        grad_logp = -p.copy()
        grad_logp[o] += 1.0                      # d log pi(o) / d logits
        ratio = p[o] / p_old[o]
        clipped = (a > 0 and ratio > 1 + eps_clip) or (a < 0 and ratio < 1 - eps_clip)
        if not clipped:                          # clipped branch has zero gradient
            g += a * ratio * grad_logp
        g -= beta * (1.0 - p_ref[o] / p[o]) * grad_logp   # d k3 / d logits
    return g / len(samples)


def train(K=8, correct=3, G=16, steps=60, inner=2, lr=0.5, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.3, K)
    logits[correct] -= 1.0                       # the base model rarely answers right
    logits_ref = logits.copy()                   # frozen reference policy
    history = []
    for step in range(steps):
        logits_old = logits.copy()
        samples = rng.choice(K, size=G, p=softmax(logits_old))
        rewards = (samples == correct).astype(float)      # rule-based reward
        adv = group_advantages(rewards)
        for _ in range(inner):                   # a few updates per sampled group
            logits = logits + lr * grpo_grad(logits, logits_old, logits_ref, samples, adv)
        pc = softmax(logits)[correct]
        history.append(pc)
        if verbose and (step % 10 == 0 or step == steps - 1):
            print(f"step {step:3d}  group accuracy {rewards.mean():.2f}  P(correct) {pc:.3f}")
    return history


if __name__ == "__main__":
    print("Advantages of a mixed group :", np.round(group_advantages([1, 0, 0, 1, 0, 0]), 2))
    print("Advantages of a uniform group:", np.round(group_advantages([1, 1, 1, 1]), 2),
          "-> no learning signal")
    p_t, p_r = np.array([0.5, 0.2]), np.array([0.4, 0.4])
    print("k3 KL estimates (always >= 0):", np.round(kl_k3(p_t, p_r), 4))
    print()
    h = train()
    print(f"\nP(correct) rose from {h[0]:.3f} to {h[-1]:.3f} without any value model.")
