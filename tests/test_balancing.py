"""Auxiliary-loss-free balancing reduces MaxVio (Chapter 6)."""
import torch

from conftest import small_config
from tinyseek.model import MoE, max_violation


def run(update: bool, steps: int = 300) -> float:
    torch.manual_seed(0)
    cfg = small_config(n_routed_experts=8, bias_update_speed=0.01)
    moe = MoE(cfg).train()
    with torch.no_grad():
        moe.gate.weight[:2] *= 6.0                      # two "loud" experts
    vio = []
    for step in range(steps):
        x = torch.randn(4, 64, cfg.hidden_size) + 0.5
        with torch.no_grad():
            moe(x)
        if update:
            moe.update_bias()
        else:
            moe.load_accum.zero_()
        if step >= steps - 50:
            vio.append(max_violation(moe.last_load))
    return sum(vio) / len(vio)


def test_bias_balancing_reduces_maxvio():
    before, after = run(False), run(True)
    assert before > 0.5
    assert after < 0.5 * before


def test_update_rule_sign():
    moe = MoE(small_config())
    moe.update_bias(torch.tensor([10.0, 0.0, 5.0, 5.0]))
    g = moe.gamma
    assert torch.allclose(moe.bias, torch.tensor([-g, g, 0.0, 0.0]))
