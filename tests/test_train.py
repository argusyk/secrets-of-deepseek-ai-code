"""Training loop and the V3 learning-rate schedule (Chapter 9)."""
import numpy as np

from conftest import small_config
from tinyseek.data import PackedDataset
from tinyseek.train import lr_schedule, train


def test_lr_schedule_shape():
    total, peak = 1000, 1e-3
    lrs = [lr_schedule(s, total, peak, warmup=50) for s in range(total)]
    assert lrs[0] < lrs[49] == peak                    # warm-up
    assert lrs[300] == lrs[600] == peak                # constant phase
    assert lrs[900] < peak and abs(lrs[970] - 0.1 * peak) < 1e-12
    assert lrs[-1] < 0.05 * peak                       # final 7.3e-6 / 2.2e-4


def test_few_steps_reduce_loss_and_balance():
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 300, 37)
    tokens = np.tile(pattern, 200)                     # a learnable periodic stream
    ds = PackedDataset(tokens, 32)
    cfg = small_config()
    model, hist = train(cfg, ds, steps=60, batch_size=8, lr=1e-2, warmup=5,
                        log_every=5, verbose=False)
    assert hist[-1]["loss"] < 0.6 * hist[0]["loss"]
    assert hist[-1]["maxvio"] is not None
    assert all(abs(float(m.bias.abs().max())) > 0 for m in model.moe_layers())
