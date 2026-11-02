"""Model configuration of TinySeek (Chapters 2 and 8).

`TinySeekConfig` mirrors the field names of DeepSeek-V3's ``config.json`` so
that a reader can put the two files side by side: ``kv_lora_rank`` is the
latent width d_c of MLA (Chapter 4), ``n_routed_experts`` and
``num_experts_per_tok`` describe the mixture of experts (Chapter 5),
``num_nextn_predict_layers`` is the MTP depth (Chapter 7) and so on. A few
TinySeek-only fields (``attention``, ``balance``, ``mtp_lambda``) switch the
DeepSeek ideas on and off for the experiments of the labs.

Four presets grow the model step by step through the book:

    tiny_dense  MHA + dense SwiGLU                 (Chapter 2 baseline)
    tiny_mla    MLA + dense SwiGLU                 (Chapter 4)
    tiny_moe    MHA + DeepSeekMoE + bias balancing (Chapters 5-6)
    tiny_v3     MLA + DeepSeekMoE + bias + MTP     (Chapters 7-8)

Run ``python -m tinyseek.config --preset tiny_v3`` to print a preset and its
parameter count.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TinySeekConfig:
    """Hyper-parameters of a TinySeek model (names follow DeepSeek-V3)."""

    vocab_size: int = 2048
    hidden_size: int = 128                 # d
    num_hidden_layers: int = 4             # L
    num_attention_heads: int = 4           # n_h
    num_key_value_heads: Optional[int] = None  # MHA only: < n_h gives GQA/MQA
    attention: str = "mla"                 # "mha" | "mla"
    q_lora_rank: int = 48                  # d_c' (query compression), 0 = off
    kv_lora_rank: int = 32                 # d_c  (cached KV latent)
    qk_nope_head_dim: int = 32             # d_h  (content part of q/k)
    qk_rope_head_dim: int = 16             # d_h^R (decoupled RoPE part)
    v_head_dim: int = 32
    rope_theta: float = 10000.0
    first_k_dense_replace: int = 1         # the first k layers keep a dense FFN
    n_routed_experts: int = 8              # N_r (0 = dense model)
    n_shared_experts: int = 1              # N_s
    num_experts_per_tok: int = 2           # K_r
    moe_intermediate_size: int = 64        # width of one fine-grained expert
    intermediate_size: int = 256           # width of a dense SwiGLU FFN
    scoring_func: str = "sigmoid"          # "softmax" (V2) | "sigmoid" (V3)
    balance: str = "bias"                  # "none" | "aux" | "bias"
    bias_update_speed: float = 1e-3        # gamma, formula (6.5)
    aux_loss_alpha: float = 1e-2           # alpha of the auxiliary loss (6.6)
    seq_aux_alpha: float = 1e-4            # tiny sequence-wise loss used with "bias"
    num_nextn_predict_layers: int = 1      # MTP depth D (0 = off)
    mtp_lambda: float = 0.3                # lambda, formula (7.6)
    max_position_embeddings: int = 256     # context length T
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False      # DeepSeek-V3 keeps them separate
    name: str = field(default="custom", compare=False)

    # ------------------------------------------------------------------ checks
    def __post_init__(self) -> None:
        if self.attention not in ("mha", "mla"):
            raise ValueError(f"attention must be 'mha' or 'mla', got {self.attention!r}")
        if self.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(f"unknown scoring_func {self.scoring_func!r}")
        if self.balance not in ("none", "aux", "bias"):
            raise ValueError(f"unknown balance {self.balance!r}")
        if self.qk_rope_head_dim % 2 or self.qk_nope_head_dim % 2:
            raise ValueError("RoPE dimensions must be even")
        if self.n_routed_experts and self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
        kv = self.kv_heads
        if self.num_attention_heads % kv:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    # --------------------------------------------------------------- helpers
    @property
    def kv_heads(self) -> int:
        """Number of key/value heads of MHA/GQA (ignored by MLA)."""
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def mha_head_dim(self) -> int:
        """Head size of plain MHA: we reuse d_h = qk_nope_head_dim."""
        return self.qk_nope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        """DeepSeek-V3 rule: layers >= first_k_dense_replace use MoE."""
        return self.n_routed_experts > 0 and layer_idx >= self.first_k_dense_replace

    def replace(self, **kw) -> "TinySeekConfig":
        """A copy with some fields changed (``dataclasses.replace``)."""
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------ parameters
    def num_parameters(self, active: bool = False, include_mtp: bool = False) -> int:
        """Analytic parameter count with the formulas of Chapter 8.

        ``active=True`` counts what one token touches: every non-expert matrix,
        the shared experts and only K_r of the N_r routed experts. The result
        equals ``model.count_parameters(model, active)`` exactly (tested).
        """
        d, h = self.hidden_size, self.num_attention_heads
        emb = self.vocab_size * d
        head = 0 if self.tie_word_embeddings else self.vocab_size * d

        def attn() -> int:
            if self.attention == "mha":
                dh, kvh = self.mha_head_dim, self.kv_heads
                return d * h * dh + 2 * d * kvh * dh + h * dh * d
            qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            if self.q_lora_rank:
                q = d * self.q_lora_rank + self.q_lora_rank + self.q_lora_rank * h * qk
            else:
                q = d * h * qk
            kv = (d * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank
                  + self.kv_lora_rank * h * (self.qk_nope_head_dim + self.v_head_dim))
            return q + kv + h * self.v_head_dim * d

        def ffn(layer_idx: int) -> int:
            if not self.is_moe_layer(layer_idx):
                return 3 * d * self.intermediate_size
            per = 3 * d * self.moe_intermediate_size
            routed = self.num_experts_per_tok if active else self.n_routed_experts
            return (self.n_shared_experts + routed) * per + self.n_routed_experts * d

        def block(layer_idx: int) -> int:
            return attn() + ffn(layer_idx) + 2 * d          # + two RMSNorm gains

        total = emb + head + d + sum(block(i) for i in range(self.num_hidden_layers))
        if include_mtp:
            per_mtp = 2 * d + 2 * d * d + block(self.num_hidden_layers) + d
            total += self.num_nextn_predict_layers * per_mtp
        return total

    # ------------------------------------------------------------- JSON I/O
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: Optional[str | Path] = None) -> str:
        """Serialize to JSON (and write to ``path`` if given)."""
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, src: str | Path | dict) -> "TinySeekConfig":
        """Load from a path, a JSON string or a dict; unknown keys are ignored
        so that a real DeepSeek ``config.json`` can be read for comparison."""
        if isinstance(src, dict):
            data = src
        else:
            s = str(src)
            data = json.loads(s) if s.lstrip().startswith("{") else json.loads(
                Path(s).read_text(encoding="utf-8"))
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _tiny_v3() -> TinySeekConfig:
    return TinySeekConfig(name="tiny_v3")


presets: dict[str, TinySeekConfig] = {
    "tiny_dense": TinySeekConfig(
        name="tiny_dense", attention="mha", n_routed_experts=0, n_shared_experts=0,
        first_k_dense_replace=4, balance="none", num_nextn_predict_layers=0),
    "tiny_mla": TinySeekConfig(
        name="tiny_mla", attention="mla", n_routed_experts=0, n_shared_experts=0,
        first_k_dense_replace=4, balance="none", num_nextn_predict_layers=0),
    "tiny_moe": TinySeekConfig(
        name="tiny_moe", attention="mha", balance="bias", num_nextn_predict_layers=0),
    "tiny_v3": _tiny_v3(),
}


def get_preset(name: str, **overrides) -> TinySeekConfig:
    """Return a fresh copy of a preset, optionally with some fields replaced."""
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(presets)}")
    return presets[name].replace(**overrides)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Print a TinySeek preset and its size "
                                "(Chapter 8).")
    p.add_argument("--preset", default="tiny_v3", choices=sorted(presets))
    p.add_argument("--out", help="write the configuration to this JSON file")
    args = p.parse_args(argv)
    cfg = get_preset(args.preset)
    print(cfg.to_json(args.out))
    print(f"parameters: total {cfg.num_parameters():,}  "
          f"active {cfg.num_parameters(active=True):,}  "
          f"with MTP {cfg.num_parameters(include_mtp=True):,}")


if __name__ == "__main__":
    main()
