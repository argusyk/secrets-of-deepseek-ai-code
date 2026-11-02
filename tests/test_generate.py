"""Generation: cache, sampling filters, speculative decoding, INT8 (Chapters 2, 7, 15)."""
import torch

from conftest import small_config
from tinyseek.generate import (filter_logits, generate_ids, quantize_int8, speculative_accept,
                               speculative_generate, weight_bytes)
from tinyseek.model import TinySeekModel


def trained_like(seed=0):
    """A random model with sharper logits so that greedy ties cannot occur."""
    torch.manual_seed(seed)
    m = TinySeekModel(small_config(initializer_range=0.2)).eval()
    return m


def test_cached_and_uncached_generation_agree():
    m = trained_like()
    ids = torch.randint(0, 300, (2, 8))
    a, la = generate_ids(m, ids, 12, temperature=0, use_cache=True, return_logits=True)
    b, lb = generate_ids(m, ids, 12, temperature=0, use_cache=False, return_logits=True)
    assert torch.equal(a, b)
    assert max(float((x - y).abs().max()) for x, y in zip(la, lb)) < 1e-4


def test_top_k_top_p_filters():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    assert torch.isinf(filter_logits(logits, top_k=2)).sum() == 2
    kept = ~torch.isinf(filter_logits(logits, top_p=0.7))
    assert kept.tolist() == [[True, True, False, False]]


def test_speculative_greedy_equals_greedy():
    for seed in range(3):
        m = trained_like(seed)
        prompt = torch.randint(0, 300, (1, 6))
        ref = generate_ids(m, prompt, 20, temperature=0)[0].tolist()
        spec, st = speculative_generate(m, prompt[0], 20, temperature=0)
        assert spec == ref
        assert st["main_steps"] <= 21 and 0 <= st["acceptance"] <= 1


def test_speculative_accept_preserves_distribution():
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.25, 0.15, 0.10])
    q = torch.tensor([0.10, 0.20, 0.30, 0.40])
    g = torch.Generator().manual_seed(1)
    n, counts, acc = 20000, torch.zeros(4), 0
    for _ in range(n):
        d = int(torch.multinomial(q, 1, generator=g))
        ok, t = speculative_accept(p, q, d, g)
        counts[t] += 1
        acc += ok
    assert (counts / n - p).abs().max() < 0.015
    assert abs(acc / n - float(torch.minimum(p, q).sum())) < 0.015


def test_speculative_sampling_runs_and_accepts_with_mtp_copy():
    m = trained_like()
    prompt = torch.randint(0, 300, (8,))
    new, st = speculative_generate(m, prompt, 16, temperature=1.0,
                                   generator=torch.Generator().manual_seed(0))
    assert len(new) == 16 and st["drafted"] > 0


def test_int8_quantization_small_error_and_smaller():
    m = trained_like()
    ids = torch.randint(0, 300, (2, 16))
    with torch.no_grad():
        ref, _ = m(ids)
        size = weight_bytes(m)
        quantize_int8(m)
        q, _ = m(ids)
    assert weight_bytes(m) < 0.5 * size
    assert float((q - ref).norm() / ref.norm()) < 0.05
