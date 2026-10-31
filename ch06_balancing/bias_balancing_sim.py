"""Auxiliary-loss-free load balancing vs. an auxiliary loss: a toy simulation.

Chapter 6 of "Secrets of DeepSeek AI". A router with N experts sends each
token to its top-K experts. Tokens come from 8 topical clusters. The router's
expert embeddings are trained to raise the affinity of the chosen experts
(a stand-in for the language-modelling objective). On its own this drives
routing collapse: a few "loud" experts attract more tokens, receive more
updates and become even more popular.

Strategies compared on identical data:

  none  - no balancing;
  aux   - an auxiliary balance loss  alpha * N * sum_i f_i * P_i  whose gradient
          is added to the router (Switch/GShard style);
  bias  - DeepSeek-V3 style: a per-expert bias b_i is added to the affinity
          ONLY when choosing the top-K experts; it moves by +/- gamma according
          to the observed load and never receives a gradient.

Metrics, averaged over the last TAIL steps:
  MaxVio = (max_i load_i - mean load) / mean load       (lower = more balanced)
  purity = share of each expert's tokens that come from its dominant cluster,
           load-weighted (higher = more specialised experts)

Run:
    python bias_balancing_sim.py
"""
import numpy as np

N, K, D = 16, 2, 16          # experts, active experts per token, feature dim
C = 8                        # topical clusters of tokens
T = 2048                     # tokens per batch
STEPS, TAIL = 600, 150
LR = 0.05                    # router learning rate


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def topk_mask(scores, k):
    """Boolean (T, N) mask of the k largest entries in each row."""
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    m = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def max_violation(mask):
    load = mask.sum(axis=0).astype(float)
    return (load.max() - load.mean()) / load.mean()


def purity(mask, cluster):
    counts = np.stack([mask[cluster == c].sum(axis=0) for c in range(C)], axis=1)
    return counts.max(axis=1).sum() / counts.sum()


def v3_route(s, b, k):
    """DeepSeek-V3 gating: select by s + b, weight by the raw s (normalised)."""
    mask = topk_mask(s + b, k)
    g = s * mask
    return mask, g / g.sum(axis=1, keepdims=True)


def update_bias(b, mask, gamma):
    """Overloaded experts get a lower bias, underloaded ones a higher one."""
    load = mask.sum(axis=0)
    return b + gamma * np.sign(load.mean() - load)


def run(strategy, alpha=0.0, gamma=0.002, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((C, D))
    E = rng.standard_normal((N, D)) * 0.3
    E[:3] *= 4.0                          # three experts start out "loud"
    b = np.zeros(N)
    vio, pur = [], []
    for step in range(STEPS):
        cluster = rng.integers(0, C, T)
        X = centers[cluster] + 0.5 * rng.standard_normal((T, D))
        s = sigmoid(X @ E.T)              # affinities (V3 uses a sigmoid)
        if strategy == "bias":
            mask, _ = v3_route(s, b, K)
        else:
            mask = topk_mask(s, K)

        g = mask * s * (1 - s) / T        # ascent on the chosen affinities
        if strategy == "aux":
            f = mask.mean(axis=0) * N / K           # load fraction
            S = s.sum(axis=1, keepdims=True)
            sp = s / S                              # normalised affinities
            ds = (f - (sp * f).sum(axis=1, keepdims=True)) / S
            g = g - alpha * N * ds * s * (1 - s) / T  # descent on aux loss
        E += LR * g.T @ X

        if strategy == "bias":                       # no gradient, just a rule
            b = update_bias(b, mask, gamma)

        if step >= STEPS - TAIL:
            vio.append(max_violation(mask))
            pur.append(purity(mask, cluster))
    return np.mean(vio), np.mean(pur)


if __name__ == "__main__":
    print(f"{'strategy':<26}{'MaxVio':>9}{'purity':>9}")
    for name, strat, a in [("no balancing", "none", 0.0),
                           ("aux loss, alpha=0.1", "aux", 0.1),
                           ("aux loss, alpha=1", "aux", 1.0),
                           ("aux loss, alpha=10", "aux", 10.0),
                           ("bias, gamma=0.002", "bias", 0.0)]:
        v, p = run(strat, a)
        print(f"{name:<26}{v:>9.3f}{p:>9.3f}")
    print("\nsensitivity of the bias method to gamma:")
    for gm in (0.0005, 0.002, 0.01, 0.05):
        v, p = run("bias", gamma=gm)
        print(f"  gamma={gm:<8}MaxVio={v:.3f}  purity={p:.3f}")
