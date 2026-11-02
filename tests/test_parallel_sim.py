"""Pipeline schedules on virtual accelerators (Chapter 11)."""
import pytest
import torch

from tinyseek.parallel_sim import (all_to_all_bytes, bubble_formula, compare_schedules,
                                   schedule_1f1b, schedule_dualpipe, simulate)


@pytest.mark.parametrize("p,m", [(4, 8), (4, 16), (8, 16), (8, 32)])
def test_simulated_bubbles_match_v3_formulas(p, m):
    F, B, W = 1.0, 2.0, 1.0
    r1 = simulate(schedule_1f1b(p, m), p, F, B, W)
    assert r1["makespan"] - m * (F + B) == pytest.approx(bubble_formula("1F1B", p, F, B))
    r2 = simulate(schedule_dualpipe(p, m), p, F, B, W)
    assert r2["makespan"] - m * (F + B) == pytest.approx(
        bubble_formula("DualPipe", p, F, B, W))
    assert r2["makespan"] < r1["makespan"]


def test_dualpipe_hides_communication():
    res = compare_schedules(4, 16, 1.0, 2.0, 1.0, comm=0.5)
    assert res["DualPipe"]["bubble_share"] < res["1F1B"]["bubble_share"]


def test_dualpipe_counts_every_chunk():
    orders = schedule_dualpipe(4, 8)
    for ops in orders:
        flat = []
        for op in ops:
            flat += list(op[1:]) if op[0] == "FB" else [op]
        assert sum(o[0] == "F" for o in flat) == 8
        assert sum(o[0] == "B" for o in flat) == 8


def test_all_to_all_dedup():
    topk = torch.tensor([[0, 1], [2, 3], [0, 3], [1, 2]])
    r = all_to_all_bytes(topk, n_experts=4, ep=2, hidden=128)
    assert r["copies_dedup"] <= r["copies_naive"]
    assert r["bytes_dedup"] <= r["bytes_naive"]
