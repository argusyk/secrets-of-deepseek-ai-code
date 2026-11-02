"""Shared fixtures for the TinySeek and chapter-example tests.

Run from the ``code/`` directory:  python -m pytest -q
"""
import sys
from pathlib import Path

import pytest
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tinyseek.config import TinySeekConfig  # noqa: E402
from tinyseek.tokenizer import ByteBPETokenizer  # noqa: E402

SAMPLE_TEXT = (
    "Модель DeepSeek-V3 стискає ключі та значення в латентний вектор. "
    "Суміш експертів активує лише кілька експертів на токен.\n\n"
    "DeepSeek-V3 compresses keys and values into a latent vector. "
    "A mixture of experts activates only a few experts per token.\n\n"
    "def add(a, b):\n    return a + b\n\n\nclass Cache:\n    def __init__(self):\n"
    "        self.items = {}\n"
    "Q: 47+38=?\nA: <think>7+8=15 4+3+1=8</think><answer>85</answer>\n"
)


def small_config(**kw) -> TinySeekConfig:
    """A model small enough for millisecond forward passes."""
    base = dict(vocab_size=300, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                attention="mla", q_lora_rank=12, kv_lora_rank=8, qk_nope_head_dim=8,
                qk_rope_head_dim=4, v_head_dim=8, first_k_dense_replace=1,
                n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=48, num_nextn_predict_layers=1,
                max_position_embeddings=64, name="test")
    base.update(kw)
    return TinySeekConfig(**base)


@pytest.fixture
def cfg():
    return small_config()


@pytest.fixture(scope="session")
def tok():
    return ByteBPETokenizer().train([SAMPLE_TEXT * 3], vocab_size=300)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
