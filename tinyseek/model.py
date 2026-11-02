"""The TinySeek model: a miniature DeepSeek-V3 in PyTorch (Chapters 2, 4-8).

Building blocks, each matching a section of the book:

* `RMSNorm`, `SwiGLU`, `apply_rope`            - the pre-norm decoder of Chapter 2
  (formulas (2.9)-(2.11)); RoPE frequencies theta_j = 10000^(-2j/d);
* `MultiHeadAttention`                          - MHA / GQA / MQA with a KV cache;
* `MultiHeadLatentAttention`                    - MLA of Chapter 4: a cached latent
  c_KV (4.2) plus a decoupled RoPE key (4.8); with ``absorbed=True`` W_UK is
  folded into the query and W_UV into the output (4.5)-(4.6) so attention runs
  directly over the cached latent;
* `MoE`                                         - DeepSeekMoE of Chapter 5: shared +
  fine-grained routed experts, softmax (V2) or sigmoid (V3) affinities,
  balancing by an auxiliary loss (6.6)-(6.7) or by a per-expert bias (6.3)-(6.5);
* `MTPModule`                                   - one sequential multi-token
  prediction module (7.2)-(7.4);
* `TinySeekModel`                               - everything together.

``TinySeekModel.forward(ids, targets)`` returns ``(logits, losses)`` with
``losses = {"lm", "mtp", "aux", "total"}``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinySeekConfig

IGNORE_INDEX = -100
Cache = list  # one dict per layer, filled in place during generation


# --------------------------------------------------------------------- basics
class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g  (formula (2.9))."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding over the last dimension of ``x``.

    ``x`` has shape (..., T, d) with even d, ``pos`` holds the T absolute
    positions. Pairs (x_j, x_{j+d/2}) are rotated by the angle pos * theta_j
    with theta_j = theta^(-2j/d) (Section 2.4; the half-split layout of
    ``ch04_mla/mla_numpy.py``).
    """
    d = x.shape[-1]
    half = d // 2
    freqs = theta ** (-torch.arange(half, dtype=torch.float64) * 2.0 / d)
    ang = pos.to(torch.float64)[:, None] * freqs[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _causal_mask(tq: int, tk: int, device=None) -> Optional[torch.Tensor]:
    """Boolean mask (True = may attend) for tq new queries after tk - tq cached keys."""
    if tq == 1:
        return None
    return torch.ones(tq, tk, dtype=torch.bool, device=device).tril(diagonal=tk - tq)


class SwiGLU(nn.Module):
    """FFN(x) = (SiLU(x W1) * x W3) W2  (formula (2.11))."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ------------------------------------------------------------------ attention
class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with RoPE and a KV cache.

    With ``num_key_value_heads < num_attention_heads`` it becomes GQA, and
    with one KV head - MQA (Section 4.2). The cache stores full keys and
    values: 2 * n_kv * d_h numbers per token and layer (formula (4.1)).
    """

    def __init__(self, cfg: TinySeekConfig) -> None:
        super().__init__()
        self.n_h, self.n_kv, self.d_h = cfg.num_attention_heads, cfg.kv_heads, cfg.mha_head_dim
        self.theta = cfg.rope_theta
        d = cfg.hidden_size
        self.wq = nn.Linear(d, self.n_h * self.d_h, bias=False)
        self.wk = nn.Linear(d, self.n_kv * self.d_h, bias=False)
        self.wv = nn.Linear(d, self.n_kv * self.d_h, bias=False)
        self.wo = nn.Linear(self.n_h * self.d_h, d, bias=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                cache: Optional[dict] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)
        q, k = apply_rope(q, pos, self.theta), apply_rope(k, pos, self.theta)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if self.n_kv != self.n_h:
            rep = self.n_h // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        mask = _causal_mask(T, k.shape[2], x.device)
        if mask is not None and T == k.shape[2]:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(o.transpose(1, 2).reshape(B, T, self.n_h * self.d_h))


class MultiHeadLatentAttention(nn.Module):
    """Multi-head latent attention (Chapter 4, formulas (4.2)-(4.9)).

    Only ``c_kv`` (d_c numbers) and the shared RoPE key ``k_rope`` (d_h^R
    numbers) are cached per token and layer. In the naive form per-head keys
    and values are materialized from the latent; in the absorbed form
    (``absorbed=True``) the query is mapped into latent space with W_UK and
    W_UV is applied after attention, so full keys/values never exist.
    Both forms give the same outputs up to rounding (tested).
    """

    def __init__(self, cfg: TinySeekConfig, absorbed: bool = False) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.n_h = cfg.num_attention_heads
        self.d_c, self.d_cq = cfg.kv_lora_rank, cfg.q_lora_rank
        self.d_h, self.d_r, self.d_v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        self.theta = cfg.rope_theta
        self.absorbed = absorbed
        qk = self.d_h + self.d_r
        if self.d_cq:
            self.wq_a = nn.Linear(d, self.d_cq, bias=False)          # W^DQ
            self.q_norm = RMSNorm(self.d_cq, cfg.rms_norm_eps)
            self.wq_b = nn.Linear(self.d_cq, self.n_h * qk, bias=False)  # [W^UQ; W^QR]
        else:
            self.wq = nn.Linear(d, self.n_h * qk, bias=False)
        self.wkv_a = nn.Linear(d, self.d_c + self.d_r, bias=False)    # [W^DKV; W^KR]
        self.kv_norm = RMSNorm(self.d_c, cfg.rms_norm_eps)
        self.wkv_b = nn.Linear(self.d_c, self.n_h * (self.d_h + self.d_v), bias=False)  # [W^UK; W^UV]
        self.wo = nn.Linear(self.n_h * self.d_v, d, bias=False)
        self.scale = 1.0 / math.sqrt(qk)

    def _queries(self, x: torch.Tensor, pos: torch.Tensor):
        B, T, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))) if self.d_cq else self.wq(x)
        q = q.view(B, T, self.n_h, self.d_h + self.d_r)
        q_nope, q_rope = q[..., :self.d_h], q[..., self.d_h:]
        q_rope = apply_rope(q_rope.transpose(1, 2), pos, self.theta)  # (B, H, T, d_r)
        return q_nope, q_rope

    def forward(self, x: torch.Tensor, pos: torch.Tensor, cache: Optional[dict] = None,
                absorbed: Optional[bool] = None) -> torch.Tensor:
        B, T, _ = x.shape
        absorbed = self.absorbed if absorbed is None else absorbed
        q_nope, q_rope = self._queries(x, pos)
        kv = self.wkv_a(x)
        c_kv = self.kv_norm(kv[..., :self.d_c])                         # (B, T, d_c)
        k_rope = apply_rope(kv[..., self.d_c:], pos, self.theta)        # (B, T, d_r)
        if cache is not None:
            if "c_kv" in cache:
                c_kv = torch.cat([cache["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([cache["k_rope"], k_rope], dim=1)
            cache["c_kv"], cache["k_rope"] = c_kv, k_rope
        S = c_kv.shape[1]
        mask = _causal_mask(T, S, x.device)
        if not absorbed:
            kvb = self.wkv_b(c_kv).view(B, S, self.n_h, self.d_h + self.d_v)
            k_nope, v = kvb[..., :self.d_h], kvb[..., self.d_h:]
            k = torch.cat([k_nope.transpose(1, 2),
                           k_rope[:, None].expand(B, self.n_h, S, self.d_r)], dim=-1)
            q = torch.cat([q_nope.transpose(1, 2), q_rope], dim=-1)
            causal = mask is not None and T == S
            o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2),
                                               attn_mask=None if causal else mask,
                                               is_causal=causal, scale=self.scale)
            o = o.transpose(1, 2)                                       # (B, T, H, d_v)
        else:
            w = self.wkv_b.weight.view(self.n_h, self.d_h + self.d_v, self.d_c)
            w_uk, w_uv = w[:, :self.d_h], w[:, self.d_h:]               # (H, d_h|d_v, d_c)
            q_lat = torch.einsum("bthd,hdc->bhtc", q_nope, w_uk)        # absorb W_UK
            scores = (torch.einsum("bhtc,bsc->bhts", q_lat, c_kv)
                      + torch.einsum("bhtr,bsr->bhts", q_rope, k_rope)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = scores.softmax(dim=-1)
            o_lat = torch.einsum("bhts,bsc->bthc", attn, c_kv)          # mix latents
            o = torch.einsum("bthc,hvc->bthv", o_lat, w_uv)             # absorb W_UV
        return self.wo(o.reshape(B, T, self.n_h * self.d_v))


# ------------------------------------------------------------------------ MoE
class MoE(nn.Module):
    """DeepSeekMoE layer (formula (5.5)) with V2/V3 routers and balancing.

    * ``scoring_func="sigmoid"`` (V3): s = sigmoid(u . e_i) (5.6); the top-K
      experts are selected by s + b (6.3) if ``balance="bias"``; the gating
      weights are the selected s normalized to sum to one (5.7).
    * ``scoring_func="softmax"`` (V2): s = softmax(u . e_i) (5.3), top-K by s.
    * ``balance="aux"`` adds L_Bal = alpha * sum_i f_i P_i (6.6)-(6.7).
    * ``balance="bias"`` keeps a per-expert bias outside the gradient; call
      `update_bias` once per optimizer step (formula (6.5)). A tiny
      sequence-wise auxiliary loss (``seq_aux_alpha``) is added as in V3.

    After every forward pass ``last_load`` holds the number of tokens sent
    to each routed expert, ``last_topk`` the chosen expert ids and
    ``aux_loss`` the balancing loss.
    """

    def __init__(self, cfg: TinySeekConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.n_routed, self.top_k = cfg.n_routed_experts, cfg.num_experts_per_tok
        self.scoring, self.balance = cfg.scoring_func, cfg.balance
        self.gamma = cfg.bias_update_speed
        self.alpha = cfg.aux_loss_alpha if cfg.balance == "aux" else (
            cfg.seq_aux_alpha if cfg.balance == "bias" else 0.0)
        self.gate = nn.Linear(d, self.n_routed, bias=False)            # centroids e_i
        self.register_buffer("bias", torch.zeros(self.n_routed))       # b_i, no gradient
        self.experts = nn.ModuleList(SwiGLU(d, cfg.moe_intermediate_size)
                                     for _ in range(self.n_routed))
        self.shared = (SwiGLU(d, cfg.n_shared_experts * cfg.moe_intermediate_size)
                       if cfg.n_shared_experts else None)
        self.last_load = torch.zeros(self.n_routed)
        self.load_accum = torch.zeros(self.n_routed)
        self.last_topk: Optional[torch.Tensor] = None
        self.aux_loss = torch.zeros(())

    def route(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (top-k indices, gating weights, affinities s) for tokens u (N, d)."""
        logits = self.gate(u).float()
        s = logits.sigmoid() if self.scoring == "sigmoid" else logits.softmax(-1)
        choice = s + self.bias if self.balance == "bias" else s
        idx = choice.topk(self.top_k, dim=-1).indices
        w = s.gather(-1, idx)
        if self.scoring == "sigmoid":
            w = w / w.sum(-1, keepdim=True)
        return idx, w.to(u.dtype), s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        u = x.reshape(-1, d)
        idx, w, s = self.route(u)
        out = torch.zeros_like(u)
        for e in range(self.n_routed):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, self.experts[e](u[tok]) * w[tok, slot, None])
        if self.shared is not None:
            out = out + self.shared(u)
        load = torch.bincount(idx.flatten(), minlength=self.n_routed).float()
        self.last_load = load.detach()
        self.last_topk = idx.detach()
        if self.training:
            self.load_accum = self.load_accum + self.last_load
        self.aux_loss = self._balance_loss(s, idx, B, T) if (
            self.training and self.alpha > 0) else torch.zeros((), device=x.device)
        return out.view(B, T, d)

    def _balance_loss(self, s, idx, B, T) -> torch.Tensor:
        """alpha * sum_i f_i P_i with f_i = N_r/(K_r T) * count_i, P_i = mean of
        normalized affinities (6.6)-(6.7). For "bias" it is computed per
        sequence (the complementary sequence-wise loss of V3)."""
        n, k = self.n_routed, self.top_k
        groups = B if self.balance == "bias" else 1
        s = s.view(groups, -1, n)
        idx = idx.view(groups, -1, k)
        tokens = s.shape[1]
        sel = torch.zeros_like(s).scatter_(-1, idx, 1.0)
        f = sel.sum(1) * n / (k * tokens)
        p = (s / s.sum(-1, keepdim=True)).mean(1)
        return self.alpha * (f * p).sum(-1).mean()

    @torch.no_grad()
    def update_bias(self, load: Optional[torch.Tensor] = None) -> None:
        """b_i <- b_i + gamma * sign(mean load - load_i)  (formula (6.5)).

        ``load`` defaults to the tokens accumulated since the previous call."""
        if load is None:
            load = self.load_accum
        self.bias += self.gamma * torch.sign(load.mean() - load)
        self.load_accum = torch.zeros_like(self.load_accum)


def max_violation(load: torch.Tensor) -> float:
    """MaxVio = (max load - mean load) / mean load  (formula (6.9))."""
    load = load.float()
    return float((load.max() - load.mean()) / load.mean().clamp_min(1e-9))


# --------------------------------------------------------------------- blocks
class Block(nn.Module):
    """Pre-norm decoder block (formula (2.10)) with MHA/MLA and dense FFN/MoE."""

    def __init__(self, cfg: TinySeekConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = (MultiHeadLatentAttention(cfg) if cfg.attention == "mla"
                     else MultiHeadAttention(cfg))
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = MoE(cfg) if cfg.is_moe_layer(layer_idx) else SwiGLU(
            cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, pos, cache=None):
        x = x + self.attn(self.attn_norm(x), pos, cache)
        return x + self.ffn(self.ffn_norm(x))


class MTPModule(nn.Module):
    """One sequential MTP module of depth k (Section 7.3).

    h'_i = M_k [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]   (7.2)
    h^k  = TRM_k(h')                                          (7.3)
    P^k  = OutHead(h^k)                                       (7.4)
    The embedding and the output head are shared with the main model.
    """

    def __init__(self, cfg: TinySeekConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.hnorm = RMSNorm(d, cfg.rms_norm_eps)
        self.enorm = RMSNorm(d, cfg.rms_norm_eps)
        self.proj = nn.Linear(2 * d, d, bias=False)                     # M_k
        self.block = Block(cfg, cfg.num_hidden_layers)                  # TRM_k
        self.norm = RMSNorm(d, cfg.rms_norm_eps)                        # before OutHead

    def forward(self, h_prev, emb_next, pos, cache=None):
        h = self.proj(torch.cat([self.hnorm(h_prev), self.enorm(emb_next)], dim=-1))
        return self.block(h, pos, cache)


# ---------------------------------------------------------------------- model
class TinySeekModel(nn.Module):
    """Decoder-only TinySeek (DeepSeek-V3 in miniature)."""

    def __init__(self, cfg: TinySeekConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp = nn.ModuleList(MTPModule(cfg) for _ in range(cfg.num_nextn_predict_layers))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.config.initializer_range)

    # ---------------------------------------------------------- utilities
    def moe_layers(self) -> list[MoE]:
        return [m for m in self.modules() if isinstance(m, MoE)]

    def set_absorbed(self, absorbed: bool = True) -> "TinySeekModel":
        """Switch every MLA layer to the absorbed (inference) form."""
        for m in self.modules():
            if isinstance(m, MultiHeadLatentAttention):
                m.absorbed = absorbed
        return self

    def _release_aux(self) -> None:
        """Keep only detached copies of the MoE balancing losses on the modules
        (the graph lives in the returned ``losses``), so the model stays
        copyable with ``copy.deepcopy`` after a training step."""
        for m in self.moe_layers():
            m.aux_loss = m.aux_loss.detach()

    def new_cache(self) -> dict:
        """Empty KV/latent cache for the main layers and the MTP modules."""
        return {"layers": [dict() for _ in self.layers],
                "mtp": [dict() for _ in self.mtp], "len": 0}

    # ------------------------------------------------------------- forward
    def hidden_states(self, ids: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        """Final hidden states (before the last norm) for ``ids`` (B, T).

        With a ``cache`` the tokens are appended after ``cache["len"]``
        cached positions and the cache is updated in place.
        """
        start = cache["len"] if cache is not None else 0
        T = ids.shape[1]
        pos = torch.arange(start, start + T, device=ids.device)
        x = self.embed(ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, cache["layers"][i] if cache is not None else None)
        if cache is not None:
            cache["len"] = start + T
        return x

    def logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h))

    def mtp_logits(self, k: int, h_prev: torch.Tensor, next_ids: torch.Tensor,
                   start: int = 0, cache: Optional[dict] = None):
        """Run MTP module ``k`` (0-based); returns (logits, hidden)."""
        pos = torch.arange(start, start + h_prev.shape[1], device=h_prev.device)
        mod = self.mtp[k]
        h = mod(h_prev, self.embed(next_ids), pos,
                cache["mtp"][k] if cache is not None else None)
        return self.lm_head(mod.norm(h)), h

    def forward(self, ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None):
        """Returns ``(logits, losses)``.

        ``targets`` are next-token labels aligned with ``ids`` (``-100`` =
        ignore). Losses: ``lm`` - cross-entropy of the main head (2.2);
        ``mtp`` - mean MTP loss over depths (7.5), weighted by lambda in
        ``total`` (7.6); ``aux`` - sum of MoE balancing losses.
        """
        h = self.hidden_states(ids, cache)
        logits = self.logits_from_hidden(h)
        losses: dict[str, torch.Tensor] = {}
        if targets is None:
            self._release_aux()
            return logits, losses
        V = logits.shape[-1]
        lm = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1),
                             ignore_index=IGNORE_INDEX)
        T = ids.shape[1]
        mtp_losses = []
        h_prev = h
        for k in range(len(self.mtp)):
            n = T - k - 1                       # positions with a depth-(k+1) target
            if n <= 0:
                break
            emb_ids = ids[:, k + 1:k + 1 + n]                   # t_{i+k+1}
            lg, h_prev = self.mtp_logits(k, h_prev[:, :n], emb_ids)
            tgt = targets[:, k + 1:k + 1 + n]                   # t_{i+k+2}
            mtp_losses.append(F.cross_entropy(lg.reshape(-1, V).float(), tgt.reshape(-1),
                                              ignore_index=IGNORE_INDEX))
        mtp = torch.stack(mtp_losses).mean() if mtp_losses else torch.zeros((), device=ids.device)
        aux = sum((m.aux_loss for m in self.moe_layers()), torch.zeros((), device=ids.device))
        self._release_aux()
        total = lm + self.config.mtp_lambda * mtp + aux
        losses = {"lm": lm, "mtp": mtp, "aux": aux, "total": total}
        return logits, losses


# ----------------------------------------------------------------- accounting
def count_parameters(model: nn.Module, active: bool = False, include_mtp: bool = False) -> int:
    """Number of parameters (Chapter 8).

    ``active=True`` counts the parameters one token uses: all routed experts
    except K_r of N_r per MoE layer are excluded. MTP modules are excluded
    unless ``include_mtp`` (DeepSeek-V3 reports 671B without its 14B MTP).
    """
    total = 0
    for name, p in model.named_parameters():
        if not include_mtp and name.startswith("mtp."):
            continue
        total += p.numel()
    if active:
        for name, m in model.named_modules():
            if isinstance(m, MoE) and (include_mtp or not name.startswith("mtp.")):
                per = sum(p.numel() for p in m.experts[0].parameters())
                total -= (m.n_routed - m.top_k) * per
    return total


def kv_cache_bytes(config: TinySeekConfig, seq_len: int, batch: int = 1,
                   bytes_per_elem: int = 2) -> int:
    """KV-cache size of the main layers in bytes (formulas (4.1), (4.10)).

    MHA/GQA: 2 * n_kv * d_h numbers per token and layer;
    MLA: d_c + d_h^R numbers per token and layer.
    """
    if config.attention == "mla":
        per_tok = config.kv_lora_rank + config.qk_rope_head_dim
    else:
        per_tok = 2 * config.kv_heads * config.mha_head_dim
    return per_tok * config.num_hidden_layers * seq_len * batch * bytes_per_elem


def cache_tensor_bytes(cache: dict) -> int:
    """Actual bytes held by the tensors of a generation cache (main layers)."""
    return sum(t.numel() * t.element_size() for layer in cache["layers"]
               for t in layer.values())


def main(argv=None) -> None:
    import argparse
    from .config import get_preset, presets
    p = argparse.ArgumentParser(description="Build a TinySeek model and print its "
                                "structure, parameters and KV-cache size (Chapter 8).")
    p.add_argument("--preset", default="tiny_v3", choices=sorted(presets))
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    cfg = get_preset(args.preset)
    m = TinySeekModel(cfg)
    print(m)
    print(f"parameters: total {count_parameters(m):,}, active {count_parameters(m, True):,}, "
          f"MTP {count_parameters(m, include_mtp=True) - count_parameters(m):,}")
    print(f"KV cache for {args.seq_len} tokens (BF16): "
          f"{kv_cache_bytes(cfg, args.seq_len) / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
