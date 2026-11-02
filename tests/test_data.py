"""Corpus utilities: dedup, packing, arithmetic tasks (Chapters 3, 12-13)."""
import numpy as np
import torch

from tinyseek.data import (PackedDataset, arith_completion, arith_reasoning,
                           make_arith_task, minhash_dedup, parse_arith_prompt)
from tinyseek.grpo import rule_reward


def test_minhash_finds_near_duplicates():
    docs = ["DeepSeek-V3 is a Mixture-of-Experts model with 671B total parameters.",
            "DeepSeek-V3 is a Mixture-of-Experts model with 671B total parameters!",
            "Byte-pair encoding merges the most frequent pair of adjacent symbols.",
            "Rotary embeddings rotate query and key pairs by position-dependent angles."]
    keep, pairs = minhash_dedup(docs, threshold=0.7)
    assert keep == [0, 2, 3]
    assert pairs and pairs[0][:2] == (0, 1)


def test_packed_dataset_shapes():
    ds = PackedDataset(np.arange(1000), seq_len=16)
    x, y = ds.get_batch(4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert len(ds) == (1000 - 1) // 16


def test_arith_tasks_are_correct():
    rng = np.random.default_rng(0)
    for _ in range(200):
        prompt, ans = make_arith_task(rng, 3)
        a, op, b = parse_arith_prompt(prompt)
        assert int(ans) == (a + b if op == "+" else a - b) >= 0
        assert rule_reward(arith_completion(prompt, ans), ans) == 2.0
    assert arith_reasoning(47, "+", 38) == "7+8=15 4+3+1=8"
    assert arith_reasoning(52, "-", 38) == "12-8=4 4-3=1"
