"""A toy model of the Jevons effect for LLM inference.

Chapter 20 of "Secrets of DeepSeek AI". Demand for tokens follows a
constant-elasticity curve D(p) = A * p**(-eps). If an efficiency gain eta
cuts the cost of serving a token by a factor eta and the price follows the
cost, then

    tokens demanded   D(p0/eta)        = D0 * eta**eps
    total spending    p * D            = S0 * eta**(eps - 1)
    compute demanded  D(p0/eta) / eta  = H0 * eta**(eps - 1)

so for eps > 1 cheaper inference *raises* the total compute bill. The
numbers are illustrative; real elasticities are unknown and change over time.

Run:
    python jevons_sim.py
"""


def demand(price, A=1.0, eps=1.5):
    """Tokens demanded at a given price (constant elasticity)."""
    return A * price ** (-eps)


def scenario(eta, eps, p0=1.0, A=1.0):
    """Relative tokens, spending and compute after an efficiency gain eta."""
    d0 = demand(p0, A, eps)
    p1 = p0 / eta
    d1 = demand(p1, A, eps)
    tokens = d1 / d0
    spending = (p1 * d1) / (p0 * d0)
    compute = (d1 / eta) / d0          # compute per token falls by eta
    return tokens, spending, compute


def table(etas=(1, 2, 4, 10), epss=(0.5, 1.0, 1.5, 2.0)):
    rows = []
    for eps in epss:
        for eta in etas:
            t, s, c = scenario(eta, eps)
            rows.append((eps, eta, t, s, c))
    return rows


if __name__ == "__main__":
    print(f"{'eps':>4} {'eta':>4} {'tokens':>9} {'spending':>9} {'compute':>9}")
    for eps, eta, t, s, c in table():
        print(f"{eps:>4.1f} {eta:>4d} {t:>9.2f} {s:>9.2f} {c:>9.2f}")
    # break-even: compute demand is unchanged exactly when eps == 1
    t, s, c = scenario(10, 1.0)
    assert abs(c - 1.0) < 1e-12 and abs(s - 1.0) < 1e-12
    print("eps = 1: spending and compute unchanged for any eta (break-even)")
