"""Compute-optimal allocation from a toy scaling law (NumPy only).

Chapter 9 of "Secrets of DeepSeek AI" (Listing 9.1). We generate noisy
"training runs" from a Chinchilla-style loss, formula (9.1),
L(N, D) = E + A_N / N**alpha_N + A_D / D**alpha_D [18], sweep the model size N
for several compute budgets C = 6 N D, pick the best N for each budget (an
"IsoFLOP" sweep) and fit N_opt ~ C**a by least squares. Theory, formula (9.2),
says a = alpha_D / (alpha_N + alpha_D). Two synthetic "datasets" differ only
in alpha_D: the one whose data term decays faster (the "cleaner" data) pushes
compute toward the model, the qualitative effect reported for DeepSeek LLM [6].

The last block evaluates the DeepSeek LLM hyper-parameter fits (9.4)
eta_opt = 0.3118 C**-0.125 and B_opt = 0.2920 C**0.3271 (B in tokens) [6].

Run:
    python scaling_law.py
"""
import numpy as np

E, A_N, A_D = 1.69, 406.4, 410.7      # Chinchilla-like constants [18]
ALPHA_N = 0.34


def loss(N, D, alpha_d, rng, noise=0.003):
    """Formula (9.1) with multiplicative noise."""
    clean = E + A_N / N ** ALPHA_N + A_D / D ** alpha_d
    return clean * (1 + noise * rng.standard_normal(np.shape(N)))


def isoflop_fit(alpha_d, rng):
    budgets = np.logspace(19, 23, 9)                 # compute budgets in FLOPs
    n_opt = []
    for C in budgets:
        N = np.logspace(7, 12, 400)                  # candidate model sizes
        D = C / (6 * N)                              # tokens that fit the budget
        L = loss(N, D, alpha_d, rng)
        # smooth the noisy curve with a quadratic in log N around the minimum
        i = int(np.argmin(L))
        lo, hi = max(i - 40, 0), min(i + 40, len(N))
        c2, c1, _ = np.polyfit(np.log(N[lo:hi]), L[lo:hi], 2)
        n_opt.append(np.exp(-c1 / (2 * c2)))
    a, _ = np.polyfit(np.log(budgets), np.log(n_opt), 1)
    return a


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    for name, ad in (("noisier data ", 0.28), ("cleaner data ", 0.40)):
        a = isoflop_fit(ad, rng)
        print(f"{name} alpha_D={ad:.2f}: fitted a={a:.3f}  "
              f"(theory {ad / (ALPHA_N + ad):.3f}), b=1-a={1 - a:.3f}")

    print("\nDeepSeek LLM hyper-parameter fits (dense models) [6]:")
    for C in (1e20, 1e22, 1e24, 3.3e24):
        eta = 0.3118 * C ** -0.125
        b_tok = 0.2920 * C ** 0.3271
        print(f"C={C:.1e}: lr~{eta:.2e}, batch~{b_tok / 1e6:6.1f}M tokens "
              f"(~{b_tok / 4096:,.0f} sequences of 4K)")
