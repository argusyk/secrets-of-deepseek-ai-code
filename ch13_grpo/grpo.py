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
    """A_i = (r_i - mean(r)) / std(r) inside one group, formula (13.3).
    The population std is used, as in np.std."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def kl_k3(p_theta, p_ref):
    """Per-sample k3 estimate of KL, formula (13.4) [Schulman, 2020]:
    x - log(x) - 1 with x = pi_ref / pi_theta. Always >= 0. Its value is
    an unbiased estimate of KL(pi_theta || pi_ref) only when the sample is
    drawn from pi_theta itself; GRPO samples from pi_old. Used as a loss,
    its expected gradient equals the gradient of KL(pi_ref || pi_theta)."""
    x = p_ref / p_theta
    return x - np.log(x) - 1.0


def grpo_objective(logits, logits_old, logits_ref, samples, adv,
                   eps_clip=0.2, beta=0.04):
    """GRPO surrogate (13.1)-(13.2) for one group of sampled answers."""
    p = softmax(logits)
    p_old, p_ref = softmax(logits_old), softmax(logits_ref)
    ratio = p[samples] / p_old[samples]          # rho_i, formula (13.2)
    surr = np.minimum(ratio * adv,
                      np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * adv)
    return np.mean(surr - beta * kl_k3(p[samples], p_ref[samples]))


def grpo_grad(logits, logits_old, logits_ref, samples, adv,
              eps_clip=0.2, beta=0.04):
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


def k3_gradient_check(K=6, seed=1):
    """Exact expectation over o ~ pi_theta of grad k3 vs the gradients of
    the two KL directions (softmax policy, gradients w.r.t. the logits)."""
    rng = np.random.default_rng(seed)
    z, z_ref = rng.normal(0, 1, K), rng.normal(0, 1, K)
    p, r = softmax(z), softmax(z_ref)
    eye = np.eye(K)
    g_k3 = sum(p[o] * (1.0 - r[o] / p[o]) * (eye[o] - p) for o in range(K))
    g_fwd = p - r                                 # grad KL(pi_ref || pi_theta)
    kl_rev = float(np.sum(p * np.log(p / r)))
    g_rev = p * (np.log(p / r) - kl_rev)          # grad KL(pi_theta || pi_ref)
    value = float(np.sum(p * kl_k3(p, r)))        # E[k3] under pi_theta
    return value, kl_rev, np.abs(g_k3 - g_fwd).max(), np.abs(g_k3 - g_rev).max()


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
    for g in ([1, 0, 1, 0], [1, 0, 0, 0], [1] + [0] * 15):
        a = group_advantages(g)
        print(f"rewards {g[:4]}{'...' if len(g) > 4 else ''} (G={len(g)}): "
              f"A(correct) = {a[0]:+.2f}, A(wrong) = {a[-1]:+.2f}")
    print("Advantages of a uniform group:", np.round(group_advantages([1, 1, 1, 1]), 2),
          "-> no learning signal")
    p_t, p_r = np.array([0.5, 0.2]), np.array([0.4, 0.4])
    print("k3 KL estimates (always >= 0):", np.round(kl_k3(p_t, p_r), 4))
    v, kl_rev, d_fwd, d_rev = k3_gradient_check()
    print(f"E[k3] under pi_theta = {v:.4f}, KL(pi_theta||pi_ref) = {kl_rev:.4f}")
    print(f"max |E grad k3 - grad KL(ref||theta)| = {d_fwd:.1e}")
    print(f"max |E grad k3 - grad KL(theta||ref)| = {d_rev:.1e}")
    print()
    h = train()
    print(f"\nP(correct) rose from {h[0]:.3f} to {h[-1]:.3f} without any value model.")
