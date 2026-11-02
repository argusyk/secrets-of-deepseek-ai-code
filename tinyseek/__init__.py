"""TinySeek: a miniature DeepSeek-V3 for learning (Secrets of DeepSeek AI).

The package grows chapter by chapter: tokenizer and corpus (Chapter 3), MLA
(Chapter 4), DeepSeekMoE with bias balancing (Chapters 5-6), multi-token
prediction (Chapter 7), the V3 schedule (Chapter 9), FP8 emulation
(Chapter 10), pipeline simulation (Chapter 11), SFT and GRPO (Chapters
12-13), distillation (Chapter 14), cached/speculative generation (Chapter 15)
and an OpenAI-compatible server (Chapters 16 and 18). Everything runs on a
CPU in minutes.

Submodules are imported lazily, so ``python -m tinyseek.<module>`` works
without warnings:

    >>> from tinyseek import TinySeekModel, get_preset
    >>> model = TinySeekModel(get_preset("tiny_v3"))
"""
import importlib
import os

# idle OpenMP threads sleep instead of spinning (set before torch is imported),
# so that several TinySeek processes on one CPU do not stall each other
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

__version__ = "0.1.0"

_EXPORTS = {
    "TinySeekConfig": "config", "get_preset": "config", "presets": "config",
    "TinySeekModel": "model", "count_parameters": "model", "kv_cache_bytes": "model",
    "ByteBPETokenizer": "tokenizer", "fim_transform": "tokenizer",
}
__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'tinyseek' has no attribute {name!r}")
