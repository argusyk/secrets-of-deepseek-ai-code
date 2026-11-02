"""A fine-grained mixture-of-experts layer with shared experts (DeepSeekMoE).

Teaching implementation in NumPy, forward pass only. It shows:
  * shared experts that every token passes through;
  * routed experts chosen by a top-K router;
  * the two gating variants: softmax over all experts (DeepSeek-V2 style)
    and sigmoid affinities normalized over the selected experts (V3 style);
  * why fine-grained segmentation keeps the compute constant while the
    number of possible expert combinations explodes;
  * the numerical softmax-vs-sigmoid example of Section 5.5.

Simplified on purpose: there is no node/group-limited routing (Section 6.4),
no load-balancing bias (Chapter 6) and no ``routed_scaling_factor`` (the
constant by which the released V2/V3 configs multiply the routed output:
16.0 in V2, 2.5 in V3). Listing 5.1 of the book is the ``gate`` method.

See Chapter 5 of "Secrets of DeepSeek AI".

Run:
    python moe_router.py
"""
from math import comb

import numpy as np

# DeepSeek-V3 MoE layer (config.json of the released model)
V3_D_MODEL = 7168        # hidden_size
V3_D_EXPERT = 2048       # moe_intermediate_size
V3_N_SHARED = 1          # n_shared_experts
V3_N_ROUTED = 256        # n_routed_experts
V3_TOP_K = 8             # num_experts_per_tok
V3_MOE_LAYERS = 58       # 61 layers, the first 3 are dense


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Expert:
    """A SwiGLU feed-forward network: W2 (silu(x W1) * (x W3))."""

    def __init__(self, d_model, d_ff, rng):
        s = 1.0 / np.sqrt(d_model)
        self.W1 = rng.standard_normal((d_model, d_ff)) * s
        self.W3 = rng.standard_normal((d_model, d_ff)) * s
        self.W2 = rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff)

    def __call__(self, x):
        return (silu(x @ self.W1) * (x @ self.W3)) @ self.W2

    @property
    def n_params(self):
        return self.W1.size + self.W3.size + self.W2.size


class DeepSeekMoELayer:
    def __init__(self, d_model, d_ff, n_shared, n_routed, top_k,
                 gating="sigmoid", seed=0):
        rng = np.random.default_rng(seed)
        self.shared = [Expert(d_model, d_ff, rng) for _ in range(n_shared)]
        self.routed = [Expert(d_model, d_ff, rng) for _ in range(n_routed)]
        # one centroid vector e_i per routed expert: s_i = gate(u . e_i)
        self.centroids = (rng.standard_normal((n_routed, d_model))
                          / np.sqrt(d_model))
        self.top_k = top_k
        self.gating = gating

    def gate(self, u):
        """Returns (indices of selected experts, their weights) per token."""
        logits = u @ self.centroids.T              # (tokens, N_r)
        if self.gating == "softmax":               # V2 style
            z = logits - logits.max(axis=-1, keepdims=True)
            s = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        else:                                      # V3 style
            s = 1.0 / (1.0 + np.exp(-logits))
        idx = np.argsort(-s, axis=-1)[:, :self.top_k]
        w = np.take_along_axis(s, idx, axis=-1)
        if self.gating == "sigmoid":
            w = w / w.sum(axis=-1, keepdims=True)  # normalize the selected
        return idx, w

    def __call__(self, u):
        out = u.copy()                                      # residual
        for e in self.shared:                               # always active
            out += e(u)
        idx, w = self.gate(u)
        for t in range(u.shape[0]):                         # per token
            for j in range(self.top_k):
                out[t] += w[t, j] * self.routed[idx[t, j]](u[t:t + 1])[0]
        return out, idx

    def param_counts(self):
        per = self.routed[0].n_params
        total = per * (len(self.shared) + len(self.routed))
        active = per * (len(self.shared) + self.top_k)
        return total, active


def report(name, layer):
    total, active = layer.param_counts()
    print(f"{name:<34} total FFN params {total:>9,}   active {active:>8,}")


def gating_example():
    """The eight-expert example of Section 5.5 (top-2)."""
    logits = np.array([1.2, -0.3, 2.1, 0.4, 1.9, -1.0, 0.0, 0.7])
    top = np.argsort(-logits)[:2]                  # experts 3 and 5
    p = softmax(logits)[top]                       # V2: weights as is
    s = sigmoid(logits)[top]                       # V3: then normalize
    print("Section 5.5 example, experts", (top + 1).tolist())
    print(f"  softmax (V2): {p.round(3).tolist()}  sum {p.sum():.3f}")
    print(f"  sigmoid (V3): {s.round(3).tolist()}  ->"
          f" normalized {(s / s.sum()).round(3).tolist()}")


if __name__ == "__main__":
    d_model, F = 64, 256
    tokens = np.random.default_rng(1).standard_normal((32, d_model))

    # Same active compute, three designs (cf. Section 5.3)
    classic = DeepSeekMoELayer(d_model, 4 * F, n_shared=0, n_routed=16,
                               top_k=2)
    fine = DeepSeekMoELayer(d_model, F, n_shared=0, n_routed=64, top_k=8)
    dsmoe = DeepSeekMoELayer(d_model, F, n_shared=1, n_routed=63, top_k=7)
    report("classic: 16 experts, top-2", classic)
    report("fine-grained: 64 experts, top-8", fine)
    report("DeepSeekMoE: 1 shared + 63, top-7", dsmoe)

    print("\nPossible combinations of routed experts per token:")
    print(f"  C(16, 2) = {comb(16, 2):,}")
    print(f"  C(64, 8) = {comb(64, 8):,}")
    print(f"  C(63, 7) = {comb(63, 7):,}")
    print(f"  C(256, 8) = {comb(V3_N_ROUTED, V3_TOP_K):.3e}"
          "   (DeepSeek-V3 routed experts)")

    y, idx = dsmoe(tokens)
    load = np.bincount(idx.ravel(), minlength=63)
    print("\noutput shape:", y.shape)
    print("tokens per routed expert (load):", load.tolist())

    print()
    gating_example()

    # DeepSeek-V3 scale of one MoE layer (3 matrices per SwiGLU expert)
    per_expert = 3 * V3_D_MODEL * V3_D_EXPERT
    n_layer = V3_N_SHARED + V3_N_ROUTED
    n_active = V3_N_SHARED + V3_TOP_K
    print(f"\nV3 one expert: {per_expert / 1e6:.1f}M params; "
          f"layer total {n_layer * per_expert / 1e9:.2f}B, "
          f"active {n_active * per_expert / 1e6:.0f}M; "
          f"{V3_MOE_LAYERS} MoE layers: "
          f"{V3_MOE_LAYERS * n_layer * per_expert / 1e9:.0f}B in experts")
