"""Reinforcement-learning minimum: REINFORCE, a baseline and KL divergence.

Toy implementation in NumPy for Chapter 12 of "Secrets of DeepSeek AI".
One prompt, four candidate answers; the policy is a softmax over four
logits theta. The rule-based reward is a format bonus of 1 that every
answer gets plus 1 for the correct answer (index 0). The script shows that

1. the REINFORCE estimate (r - b) * grad log pi(a) has the same mean with
   and without a baseline b, but a much smaller variance with b = E[r];
2. a few hundred policy-gradient steps raise the probability of the correct
   answer, and the KL divergence to the initial (reference) policy grows;
3. KL is asymmetric, and its sample estimate mean(log pi - log pi_ref)
   over answers drawn from pi converges to the exact value.

Run:
    python rl_minimum.py
"""
import numpy as np


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def reinforce_grad(theta, a, r, b=0.0):
    """One-sample policy-gradient estimate (r - b) * d log pi(a) / d theta.
    For a softmax policy d log pi(a) / d theta = onehot(a) - pi."""
    g = -softmax(theta)
    g[a] += 1.0
    return (r - b) * g


def kl_divergence(p, q):
    """KL(p || q) = sum_y p(y) * (log p(y) - log q(y)) for categoricals."""
    return float(np.sum(p * (np.log(p) - np.log(q))))


def grad_stats(theta, reward, b, n, rng):
    """Mean and total variance of n one-sample REINFORCE estimates."""
    pi = softmax(theta)
    acts = rng.choice(len(pi), size=n, p=pi)
    g = np.array([reinforce_grad(theta, a, reward[a], b) for a in acts])
    return g.mean(axis=0), g.var(axis=0).sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    reward = np.array([2.0, 1.0, 1.0, 1.0])        # format + accuracy
    theta = np.log(np.array([0.1, 0.3, 0.3, 0.3]))  # initial policy
    pi0 = softmax(theta)
    exact = pi0 * (reward - pi0 @ reward)           # exact gradient of E[r]
    print("exact gradient of E[r]:", np.round(exact, 3))
    baselines = (("no baseline", 0.0), ("baseline b = E[r]", pi0 @ reward))
    for name, b in baselines:
        mean, var = grad_stats(theta, reward, b, 20000, rng)
        print(f"{name:18s} mean {np.round(mean, 3)}  variance {var:.3f}")

    print("policy-gradient training with a moving-average baseline:")
    b, lr = 0.0, 0.5
    for step in range(1, 301):
        a = rng.choice(4, p=softmax(theta))
        theta += lr * reinforce_grad(theta, a, reward[a], b)
        b = 0.9 * b + 0.1 * reward[a]
        if step in (1, 50, 100, 200, 300):
            pi = softmax(theta)
            print(f"  step {step:3d}: P(correct) {pi[0]:.3f}  "
                  f"KL(pi || pi_ref) {kl_divergence(pi, pi0):.3f}")

    p = np.array([0.97, 0.01, 0.01, 0.01])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"KL(p || q) = {kl_divergence(p, q):.3f}, "
          f"KL(q || p) = {kl_divergence(q, p):.3f}")
    ys = rng.choice(4, size=5000, p=p)
    est = np.mean(np.log(p[ys]) - np.log(q[ys]))
    print(f"sample estimate of KL(p || q) from 5000 draws: {est:.3f}")
