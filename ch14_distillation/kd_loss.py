"""Knowledge-distillation losses: temperature-scaled KD and forward vs reverse KL.

Teaching code for Chapter 14 of "Secrets of DeepSeek AI". NumPy only, runs
offline. Three demonstrations:

1. Hinton-style KD loss with an analytic gradient, checked numerically.
2. Why the KL term is multiplied by tau^2: without it the gradient shrinks
   roughly as 1/tau^2 when the temperature tau grows.
3. Forward KL (what SFT on teacher traces minimizes) covers all modes of a
   bimodal teacher; reverse KL (what on-policy distillation minimizes) picks
   one mode. The student is a deliberately narrow, unimodal family.

Run:
    python kd_loss.py
"""
import numpy as np


def softmax(z, tau=1.0):
    z = np.asarray(z, dtype=float) / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def kl(p, q, eps=1e-12):
    """Mean KL(p || q) over the batch (last axis = classes)."""
    return float(np.mean(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)))


def kd_loss(student_logits, teacher_logits, labels, tau=2.0, alpha=0.5):
    """alpha * tau^2 * KL(p_tau || q_tau) + (1 - alpha) * CE(y, q), (14.2).

    Returns the loss and its gradient with respect to the student logits:
    d/dz [tau^2 KL(p_tau || q_tau)] = tau (q_tau - p_tau);  d/dz CE = q - y.
    """
    p_t = softmax(teacher_logits, tau)       # soft labels of the teacher
    q_t = softmax(student_logits, tau)       # soft predictions of the student
    q = softmax(student_logits, 1.0)
    n = q.shape[0]
    y = np.zeros_like(q)
    y[np.arange(n), labels] = 1.0
    soft = tau * tau * kl(p_t, q_t)
    hard = float(-np.mean(np.log(q[np.arange(n), labels] + 1e-12)))
    loss = alpha * soft + (1.0 - alpha) * hard
    grad = (alpha * tau * (q_t - p_t) + (1.0 - alpha) * (q - y)) / n
    return loss, grad


def numeric_grad(f, z, h=1e-6):
    g = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = z[i]
        z[i] = old + h; fp = f(z)
        z[i] = old - h; fm = f(z)
        z[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


# ---------- forward vs reverse KL on a bimodal teacher ----------

def bimodal_teacher(K=101, m1=35.0, m2=65.0, s=3.0):
    x = np.arange(K)
    p = 0.5 * np.exp(-(x - m1) ** 2 / (2 * s * s)) + 0.5 * np.exp(-(x - m2) ** 2 / (2 * s * s))
    return p / p.sum()


def gaussian_student(theta, K):
    mu, log_sigma = theta
    s = np.exp(log_sigma)
    x = np.arange(K)
    return softmax(-(x - mu) ** 2 / (2 * s * s))


def fit(objective, theta0, K, lr=0.05, steps=3000):
    """Adam on (mu, log_sigma) with central-difference gradients; sigma is kept
    inside [0.5, K] so the student stays a proper bump on the grid."""
    th = np.array(theta0, dtype=float)
    m = np.zeros_like(th); v = np.zeros_like(th)
    lo, hi = np.log(0.5), np.log(float(K))
    for t in range(1, steps + 1):
        g = np.zeros_like(th)
        for k in range(len(th)):
            e = np.zeros_like(th); e[k] = 1e-4
            g[k] = (objective(th + e) - objective(th - e)) / 2e-4
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        th -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        th[1] = np.clip(th[1], lo, hi)
    return th


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) KD loss and gradient check
    zs = rng.standard_normal((4, 6))
    zt = 2.0 * rng.standard_normal((4, 6))
    labels = zt.argmax(axis=-1)
    loss, g = kd_loss(zs, zt, labels, tau=3.0, alpha=0.7)
    g_num = numeric_grad(lambda z: kd_loss(z, zt, labels, 3.0, 0.7)[0], zs.copy())
    print(f"KD loss = {loss:.4f}; max |analytic - numeric| grad = {np.abs(g - g_num).max():.2e}")

    # 2) gradient scale with and without the tau^2 factor
    print("\ntau  |grad| without tau^2 |grad| with tau^2")
    for tau in (1, 2, 4, 8):
        diff = softmax(zs, tau) - softmax(zt, tau)
        g_plain = np.linalg.norm(diff / tau)     # d KL(p||q) / dz
        g_scaled = np.linalg.norm(tau * diff)    # d [tau^2 KL] / dz
        print(f"{tau:<4} {g_plain:>14.4f}   {g_scaled:>14.4f}")

    # 3) forward vs reverse KL with a unimodal student
    K = 101
    p = bimodal_teacher(K)
    x = np.arange(K)
    mean = float((p * x).sum()); std = float(np.sqrt((p * (x - mean) ** 2).sum()))
    fwd = fit(lambda th: kl(p[None], gaussian_student(th, K)[None]), [41.0, np.log(3.0)], K)
    rev = fit(lambda th: kl(gaussian_student(th, K)[None], p[None]), [41.0, np.log(3.0)], K)
    print("\nTeacher: two modes at positions 35 and 65 (width 3)")
    print(f"teacher mean = {mean:5.2f}, std = {std:5.2f}")
    print(f"forward KL(p||q): mu = {fwd[0]:5.2f}, sigma = {np.exp(fwd[1]):5.2f}  (spreads over both modes)")
    print(f"reverse KL(q||p): mu = {rev[0]:5.2f}, sigma = {np.exp(rev[1]):5.2f}  (locks onto one mode)")
