"""Configuration, presets and parameter accounting (Chapters 2 and 8)."""
from pathlib import Path

import pytest

from tinyseek.config import TinySeekConfig, get_preset, presets
from tinyseek.model import TinySeekModel, count_parameters

CONFIG_DIR = Path(__file__).resolve().parents[1] / "tinyseek" / "configs"


def test_presets_exist_and_roundtrip(tmp_path):
    assert set(presets) == {"tiny_dense", "tiny_mla", "tiny_moe", "tiny_v3"}
    cfg = get_preset("tiny_v3")
    path = tmp_path / "c.json"
    cfg.to_json(path)
    assert TinySeekConfig.from_json(path) == cfg
    assert TinySeekConfig.from_json(cfg.to_json()) == cfg


def test_spec_sizes_of_tiny_v3():
    c = get_preset("tiny_v3")
    assert (c.vocab_size, c.hidden_size, c.num_hidden_layers, c.num_attention_heads) == (
        2048, 128, 4, 4)
    assert (c.kv_lora_rank, c.q_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim) == (
        32, 48, 32, 16)
    assert (c.n_shared_experts, c.n_routed_experts, c.num_experts_per_tok) == (1, 8, 2)
    assert c.moe_intermediate_size == 64 and c.num_nextn_predict_layers == 1
    assert c.max_position_embeddings == 256 and c.first_k_dense_replace == 1


@pytest.mark.parametrize("name", sorted(presets))
def test_analytic_count_matches_model(name):
    cfg = get_preset(name)
    m = TinySeekModel(cfg)
    for active in (False, True):
        for mtp in (False, True):
            assert count_parameters(m, active, mtp) == cfg.num_parameters(active, mtp)


def test_deepseek_v3_config_gives_671b_and_37b():
    v3 = TinySeekConfig.from_json(CONFIG_DIR / "deepseek_v3.json")
    assert abs(v3.num_parameters() / 1e9 - 671) < 1.0
    assert abs(v3.num_parameters(active=True) / 1e9 - 37) < 1.0


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        TinySeekConfig(attention="gqa")
    with pytest.raises(ValueError):
        TinySeekConfig(balance="magic")
