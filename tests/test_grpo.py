"""GRPO pieces: advantages, KL estimator, loss, rule reward (Chapter 13)."""
import torch

from tinyseek.grpo import grpo_advantages, grpo_loss, kl_k3, rule_reward


def test_advantages_zero_mean_unit_std():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 2.0, 1.0, 1.0]])
    a = grpo_advantages(r)
    assert torch.allclose(a.mean(-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(a.std(-1, unbiased=False), torch.ones(2), atol=1e-4)
    assert torch.all(grpo_advantages(torch.ones(1, 4)) == 0)   # no signal in a uniform group


def test_kl_k3_nonnegative_and_zero_at_equality():
    lp, lr = torch.log(torch.rand(1000)), torch.log(torch.rand(1000))
    assert torch.all(kl_k3(lp, lr) >= -1e-7)
    assert torch.allclose(kl_k3(lp, lp), torch.zeros(1000))


def test_grpo_loss_gradient_direction():
    logp = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]])).requires_grad_(True)
    old = logp.detach().clone()
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 2)
    loss, st = grpo_loss(logp, old, old, adv, mask, beta=0.0)
    loss.backward()
    assert torch.all(logp.grad[0] < 0) and torch.all(logp.grad[1] > 0)  # raise good, lower bad
    assert st["kl"] == 0.0


def test_rule_reward():
    ref = "85"
    assert rule_reward("<think>7+8=15 4+3+1=8</think><answer>85</answer>", ref) == 2.0
    assert rule_reward("<think>x</think><answer>84</answer>", ref) == 1.0
    assert rule_reward("85", ref) == 0.0
    assert rule_reward("<think>x</think><answer>85.0</answer>", ref) == 2.0
