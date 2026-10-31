"""The PPO clipped surrogate objective on toy numbers.

Teaching implementation in NumPy for Chapter 12 of "Secrets of DeepSeek AI".
For each token the probability ratio rho = pi_new / pi_old is multiplied by
the advantage A, but the ratio is clipped to [1 - eps, 1 + eps], and the
smaller (more pessimistic) of the two products is taken. The policy thus
gains nothing from moving too far from the policy that collected the data.

Run:
    python ppo_clip.py
"""
import numpy as np


def ppo_objective(ratio, adv, eps=0.2):
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv
    return np.minimum(unclipped, clipped)


def kl_penalized_reward(reward, logp_pi, logp_ref, beta=0.05):
    """RLHF reward shaped by a per-token KL estimate to the reference."""
    return reward - beta * (logp_pi - logp_ref)


if __name__ == "__main__":
    ratios = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    for adv in (+1.0, -1.0):
        obj = ppo_objective(ratios, adv)
        print(f"advantage {adv:+.0f}:")
        for r, o in zip(ratios, obj):
            print(f"  ratio {r:.1f} -> objective {o:+.2f}")
    print("note: with A>0 the objective stops growing above 1+eps;")
    print("      with A<0 it stops improving below 1-eps.")

    r = kl_penalized_reward(reward=1.0, logp_pi=-1.2, logp_ref=-3.0)
    print(f"reward 1.0 with a large drift from the reference -> {r:.3f}")
