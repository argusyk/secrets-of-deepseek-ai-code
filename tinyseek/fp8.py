"""FP8 (E4M3) training emulation with fine-grained scaling (Chapter 10).

DeepSeek-V3 runs the three GEMMs of every linear layer (forward, input
gradient, weight gradient) with FP8 inputs and FP32 accumulation. Scales
are chosen per 1x128 tile of activations/gradients and per 128x128 block of
weights (formulas (10.3)-(10.5)), which confines the damage of outliers.

This module *emulates* that recipe on a CPU: tensors are rounded to the
E4M3 grid (max 448, 3 mantissa bits) after scaling and multiplied back in
FP32. Rounding uses ``torch.float8_e4m3fn`` when the installed PyTorch has
it and an explicit grid otherwise; both give identical values.

* `fake_quant_e4m3(x, block)` - quantize-dequantize with tiles (1, 128),
  blocks (128, 128) or one per-tensor scale (``block=None``);
* `FP8Linear` - a linear layer whose forward and backward GEMMs see FP8 inputs;
* `convert_to_fp8(model)` - swap ``nn.Linear`` layers in place, keeping the
  embedding, the output head and the MoE gate in high precision (Section 10.5).

CLI::

    python -m tinyseek.fp8 --steps 300 --preset tiny_v3
"""
from __future__ import annotations

import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0
_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


def e4m3_grid() -> torch.Tensor:
    """All non-negative finite E4M3 values (bias 7, no infinities, 0x7F = NaN)."""
    vals = [0.0] + [m / 8 * 2.0 ** -6 for m in range(1, 8)]             # subnormals
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            vals.append((1 + m / 8) * 2.0 ** (e - 7))
    return torch.tensor(sorted(vals), dtype=torch.float32)


_GRID = e4m3_grid()


def round_e4m3(x: torch.Tensor, use_torch_dtype: bool = True) -> torch.Tensor:
    """Round to the nearest E4M3 value (inputs must already be within +-448)."""
    x = x.float().clamp(-E4M3_MAX, E4M3_MAX)
    if use_torch_dtype and _HAS_FP8:
        return x.to(torch.float8_e4m3fn).float()
    a = x.abs()
    idx = torch.searchsorted(_GRID, a).clamp(1, len(_GRID) - 1)
    lo, hi = _GRID[idx - 1], _GRID[idx]
    # ties go to the value with an even mantissa, as in IEEE rounding
    pick_hi = (hi - a < a - lo) | ((hi - a == a - lo) & (((idx) % 2) == 0))
    return torch.sign(x) * torch.where(pick_hi, hi, lo)


def fake_quant_e4m3(x: torch.Tensor, block: Optional[tuple[int, int] | int] = (1, 128)
                    ) -> torch.Tensor:
    """x ~ s * Q(x / s) with one scale s = max|x| / 448 per block (10.3)-(10.4).

    ``block`` is (rows, cols) of one scaling group over the last two
    dimensions: (1, 128) - activation tiles, (128, 128) - weight blocks; an
    int b means (1, b); ``None`` - one scale for the whole tensor.
    """
    if block is None:
        s = x.detach().abs().max().clamp_min(1e-12) / E4M3_MAX
        return round_e4m3(x / s) * s
    if isinstance(block, int):
        block = (1, block)
    br, bc = block
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]) if x.dim() != 2 else x
    R, C = x2.shape
    pr, pc = (-R) % br, (-C) % bc
    xp = F.pad(x2, (0, pc, 0, pr))
    Rp, Cp = xp.shape
    tiles = xp.reshape(Rp // br, br, Cp // bc, bc)
    s = tiles.detach().abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / E4M3_MAX
    q = (round_e4m3(tiles / s) * s).reshape(Rp, Cp)[:R, :C]
    return q.reshape(shape).to(x.dtype)


class _FP8MatMul(torch.autograd.Function):
    """y = Q(x) Q(W)^T with Q-quantized operands in all three GEMMs."""

    @staticmethod
    def forward(ctx, x, w, act_block, w_block):
        xq = fake_quant_e4m3(x, act_block)
        wq = fake_quant_e4m3(w, w_block)
        ctx.save_for_backward(xq, wq)
        ctx.act_block = act_block
        return xq @ wq.t()

    @staticmethod
    def backward(ctx, g):
        xq, wq = ctx.saved_tensors
        gq = fake_quant_e4m3(g, ctx.act_block)                 # 1x128 tiles of dY
        grad_x = gq @ wq                                       # Dgrad
        grad_w = gq.reshape(-1, gq.shape[-1]).t() @ xq.reshape(-1, xq.shape[-1])  # Wgrad
        return grad_x, grad_w, None, None


class FP8Linear(nn.Linear):
    """``nn.Linear`` whose GEMMs use E4M3-rounded inputs and FP32 accumulation.

    The master weight stays in FP32 (as V3 keeps FP32 master weights and
    optimizer states); only the GEMM operands are quantized.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 act_block=(1, 128), w_block=(128, 128)) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.act_block, self.w_block = act_block, w_block

    @classmethod
    def from_linear(cls, lin: nn.Linear, act_block=(1, 128), w_block=(128, 128)) -> "FP8Linear":
        m = cls(lin.in_features, lin.out_features, lin.bias is not None, act_block, w_block)
        m.weight = lin.weight                                  # share the master weight
        if lin.bias is not None:
            m.bias = lin.bias
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _FP8MatMul.apply(x, self.weight, self.act_block, self.w_block)
        return y + self.bias if self.bias is not None else y


def convert_to_fp8(model: nn.Module, act_block=(1, 128), w_block=(128, 128),
                   skip: tuple[str, ...] = ("lm_head", "gate")) -> nn.Module:
    """Replace ``nn.Linear`` modules by `FP8Linear` in place (Chapter 10).

    Names containing any of ``skip`` stay in high precision; the embedding is
    not a Linear and is never converted. Returns the model.
    """
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if type(child) is nn.Linear and not any(s in full for s in skip):
                setattr(mod, child_name, FP8Linear.from_linear(child, act_block, w_block))
    return model


def convert_from_fp8(model: nn.Module) -> nn.Module:
    """Turn `FP8Linear` layers back into ordinary ``nn.Linear`` (same weights)."""
    for mod in list(model.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, FP8Linear):
                lin = nn.Linear(child.in_features, child.out_features, child.bias is not None)
                lin.weight = child.weight
                if child.bias is not None:
                    lin.bias = child.bias
                setattr(mod, child_name, lin)
    return model


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    return float((approx - exact).detach().norm() / exact.detach().norm().clamp_min(1e-12))


def main(argv=None) -> None:
    from .config import get_preset, presets
    from .data import DEFAULT_DATA_DIR, PackedDataset, prepare_data
    from .model import TinySeekModel
    from .train import evaluate_loss, set_seed, train
    p = argparse.ArgumentParser(description="Compare FP32 and emulated FP8 training "
                                "of TinySeek (Chapter 10).")
    p.add_argument("--preset", default="tiny_v3", choices=sorted(presets))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--block", type=int, default=128, help="tile width (1 x block)")
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cfg = get_preset(args.preset)
    _, data = prepare_data(args.data, cfg.vocab_size, args.seed)
    tr, va = PackedDataset(data["train"], 256), PackedDataset(data["val"], 256)
    for label, fp8 in (("fp32", False), ("fp8", True)):
        set_seed(args.seed)
        m = TinySeekModel(cfg)
        if fp8:
            convert_to_fp8(m, (1, args.block), (args.block, args.block))
        m, h = train(cfg, tr, args.steps, seed=args.seed, model=m, verbose=False)
        print(f"{label}: final train loss {h[-1]['loss']:.4f}, "
              f"val loss {evaluate_loss(m, va)['val_loss']:.4f}")


if __name__ == "__main__":
    main()
