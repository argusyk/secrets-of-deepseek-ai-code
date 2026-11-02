"""Model blocks: shapes, MLA absorption, KV cache, MoE, MTP (Chapters 2, 4-8)."""
import math

import pytest
import torch

from conftest import small_config
from tinyseek.config import get_preset
from tinyseek.model import (MoE, MultiHeadLatentAttention, TinySeekModel, apply_rope,
                            cache_tensor_bytes, count_parameters, kv_cache_bytes)


@pytest.mark.parametrize("attention", ["mha", "mla"])
def test_forward_shapes_and_losses(attention):
    cfg = small_config(attention=attention)
    m = TinySeekModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 20))
    logits, losses = m(ids, torch.randint(0, cfg.vocab_size, (3, 20)))
    assert logits.shape == (3, 20, cfg.vocab_size)
    assert set(losses) == {"lm", "mtp", "aux", "total"}
    # an untrained model sits near ln(V) for both heads (Section 7.4)
    for k in ("lm", "mtp"):
        assert abs(losses[k].item() - math.log(cfg.vocab_size)) < 0.5


def test_rope_preserves_norm_and_relative_positions():
    x = torch.randn(5, 8)
    y = apply_rope(x, torch.arange(5))
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    q, k = torch.randn(1, 8), torch.randn(1, 8)
    d1 = apply_rope(q, torch.tensor([3])) @ apply_rope(k, torch.tensor([1])).T
    d2 = apply_rope(q, torch.tensor([7])) @ apply_rope(k, torch.tensor([5])).T
    assert torch.allclose(d1, d2, atol=1e-5)


def test_absorbed_mla_equals_naive_float64():
    cfg = small_config()
    attn = MultiHeadLatentAttention(cfg).double()
    x = torch.randn(2, 11, cfg.hidden_size, dtype=torch.float64)
    pos = torch.arange(11)
    y1 = attn(x, pos, absorbed=False)
    y2 = attn(x, pos, absorbed=True)
    assert (y1 - y2).abs().max() < 1e-12


def test_absorbed_model_same_logits():
    m = TinySeekModel(get_preset("tiny_v3")).eval()
    ids = torch.randint(0, 2048, (2, 40))
    with torch.no_grad():
        a, _ = m(ids)
        b, _ = m.set_absorbed(True)(ids)
    assert (a - b).abs().max() < 1e-4


@pytest.mark.parametrize("kw", [dict(attention="mha"), dict(attention="mha", num_key_value_heads=1),
                                dict(attention="mla"), dict(attention="mla", q_lora_rank=0)])
def test_cached_logits_equal_full_forward(kw):
    cfg = small_config(**kw)
    m = TinySeekModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full, _ = m(ids)
        cache = m.new_cache()
        parts = [m(ids[:, :10], cache=cache)[0]]
        parts += [m(ids[:, t:t + 1], cache=cache)[0] for t in range(10, 24)]
    assert (torch.cat(parts, 1) - full).abs().max() < 1e-5
    assert cache_tensor_bytes(cache) == kv_cache_bytes(cfg, 24, batch=2, bytes_per_elem=4)


def test_kv_cache_formula():
    mha, mla = get_preset("tiny_dense"), get_preset("tiny_v3")
    assert kv_cache_bytes(mha, 256) == 2 * 4 * 32 * 4 * 256 * 2
    assert kv_cache_bytes(mla, 256) == (32 + 16) * 4 * 256 * 2


def test_moe_active_parameters():
    cfg = get_preset("tiny_v3")
    m = TinySeekModel(cfg)
    per_expert = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    n_moe = cfg.num_hidden_layers - cfg.first_k_dense_replace
    diff = count_parameters(m) - count_parameters(m, active=True)
    assert diff == n_moe * (cfg.n_routed_experts - cfg.num_experts_per_tok) * per_expert


def test_moe_routing_and_load():
    cfg = small_config()
    moe = MoE(cfg)
    x = torch.randn(2, 10, cfg.hidden_size)
    y = moe(x)
    assert y.shape == x.shape
    assert int(moe.last_load.sum()) == 2 * 10 * cfg.num_experts_per_tok
    idx, w, s = moe.route(x.view(-1, cfg.hidden_size))
    assert torch.allclose(w.sum(-1), torch.ones(20), atol=1e-5)   # V3 normalization (5.7)


def test_moe_bias_changes_selection_not_weights():
    cfg = small_config()
    moe = MoE(cfg)
    u = torch.randn(50, cfg.hidden_size)
    idx0, _, s0 = moe.route(u)
    moe.bias[:] = torch.tensor([10.0, 10.0, -10.0, -10.0])
    idx1, w1, s1 = moe.route(u)
    assert torch.equal(s0, s1)                       # affinities untouched
    assert set(idx1.flatten().tolist()) == {0, 1}    # selection follows s + b
    assert torch.allclose(w1, s1.gather(-1, idx1) / s1.gather(-1, idx1).sum(-1, keepdim=True))


def test_aux_loss_only_when_requested():
    for bal, expect in (("none", False), ("aux", True)):
        m = TinySeekModel(small_config(balance=bal)).train()
        ids = torch.randint(0, 300, (2, 16))
        _, losses = m(ids, ids)
        assert (losses["aux"].item() > 0) == expect


def test_mtp_loss_uses_shifted_targets():
    cfg = small_config()
    m = TinySeekModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    _, l1 = m(ids, tgt)
    tgt2 = tgt.clone()
    tgt2[:, 0] = (tgt2[:, 0] + 1) % cfg.vocab_size   # t_1 is never an MTP target
    _, l2 = m(ids, tgt2)
    assert torch.allclose(l1["mtp"], l2["mtp"])
    assert not torch.allclose(l1["lm"], l2["lm"])
    l1["total"].backward()
    assert m.mtp[0].proj.weight.grad is not None
    no_mtp = TinySeekModel(small_config(num_nextn_predict_layers=0))
    _, l3 = no_mtp(ids, tgt)
    assert l3["mtp"].item() == 0.0


def test_model_deepcopy_after_training_step():
    import copy
    m = TinySeekModel(small_config(balance="aux")).train()
    ids = torch.randint(0, 300, (2, 16))
    _, losses = m(ids, ids)
    losses["total"].backward()
    clone = copy.deepcopy(m)             # GRPO keeps a frozen reference copy
    assert count_parameters(clone) == count_parameters(m)
