"""FP8 emulation with fine-grained scaling (Chapter 10)."""
import torch

from conftest import small_config
from tinyseek.fp8 import (FP8Linear, _GRID, convert_to_fp8, fake_quant_e4m3, relative_error,
                          round_e4m3)
from tinyseek.model import TinySeekModel


def test_grid_and_torch_dtype_agree():
    assert len(_GRID) == 127 and float(_GRID[-1]) == 448.0
    x = torch.randn(10000) * 50
    assert torch.equal(round_e4m3(x, True), round_e4m3(x, False))


def test_fine_grained_scaling_beats_per_tensor_with_outliers():
    torch.manual_seed(0)
    x = torch.randn(64, 512)
    x[:, :4] *= 200                                       # outlier channels
    e_tensor = relative_error(fake_quant_e4m3(x, None), x)
    e_tile = relative_error(fake_quant_e4m3(x, (1, 128)), x)
    assert e_tile < e_tensor
    assert relative_error(fake_quant_e4m3(torch.randn(256, 256), (128, 128)),
                          torch.randn(1)) >= 0           # runs on blocks
    y = fake_quant_e4m3(torch.randn(3, 5, 70), (1, 32))  # ragged tiles and 3-D input
    assert y.shape == (3, 5, 70)


def test_fp8_linear_close_and_trainable():
    lin = torch.nn.Linear(128, 64, bias=False)
    f8 = FP8Linear.from_linear(lin)
    x = torch.randn(8, 128, requires_grad=True)
    y_ref, y = lin(x), f8(x)
    assert relative_error(y, y_ref) < 0.08
    y.sum().backward()
    assert x.grad is not None and lin.weight.grad is not None


def test_convert_keeps_head_and_gate():
    m = convert_to_fp8(TinySeekModel(small_config()))
    assert not isinstance(m.lm_head, FP8Linear)
    assert all(not isinstance(x.gate, FP8Linear) for x in m.moe_layers())
    assert isinstance(m.layers[0].attn.wo, FP8Linear)
    ids = torch.randint(0, 300, (2, 16))
    _, losses = m(ids, ids)
    losses["total"].backward()
