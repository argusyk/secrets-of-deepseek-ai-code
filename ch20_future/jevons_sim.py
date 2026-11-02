"""A toy model of the Jevons effect for LLM inference.

Chapter 20 of "Secrets of DeepSeek AI" (Listing 20.1, Table 20.1).
Demand for tokens follows a constant-elasticity curve Q(p) = a * p**(-e)
(formula (20.1); the elasticity is called e, not epsilon, so that it is
not confused with the GRPO clipping bound of Chapter 13). If an
efficiency gain eta cuts the cost of serving a token by a factor eta and
the price follows the cost, then

    tokens demanded   Q(p0/eta)        = Q0 * eta**e
    total spending    p * Q            = S0 * eta**(e - 1)      (20.2)
    compute demanded  Q(p0/eta) / eta  = H0 * eta**(e - 1)      (20.3)

so for e > 1 cheaper inference *raises* the total compute bill. The
numbers are illustrative; real elasticities are unknown and change over
time.

Run:
    python jevons_sim.py
"""


def demand(price, a=1.0, e=1.5):
    """Tokens demanded at a given price (constant elasticity e)."""
    return a * price ** (-e)


def scenario(eta, e, p0=1.0, a=1.0):
    """Relative tokens, spending and compute after an efficiency gain."""
    q0 = demand(p0, a, e)
    p1 = p0 / eta                      # the price follows the cost
    q1 = demand(p1, a, e)
    tokens = q1 / q0
    spending = (p1 * q1) / (p0 * q0)
    compute = (q1 / eta) / q0          # compute per token falls by eta
    return tokens, spending, compute


def table(etas=(1, 2, 4, 10), es=(0.5, 1.0, 1.5, 2.0)):
    rows = []
    for e in es:
        for eta in etas:
            t, s, c = scenario(eta, e)
            rows.append((e, eta, t, s, c))
    return rows


if __name__ == "__main__":
    print(f"{'e':>4} {'eta':>4} {'tokens':>9} {'spending':>9} "
          f"{'compute':>9}")
    for e, eta, t, s, c in table():
        print(f"{e:>4.1f} {eta:>4d} {t:>9.2f} {s:>9.2f} {c:>9.2f}")
    # break-even: compute demand is unchanged exactly when e == 1
    t, s, c = scenario(10, 1.0)
    assert abs(c - 1.0) < 1e-12 and abs(s - 1.0) < 1e-12
    print("e = 1: spending and compute unchanged for any eta (break-even)")
