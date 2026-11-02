"""Text generation: sampling, KV/latent cache, speculative decoding, INT8
(Chapters 2, 7 and 15).

* `sample` - temperature, top-k and top-p (nucleus) sampling (Chapter 2);
  greedy decoding is ``temperature=0``.
* `generate_ids` - batched generation with or without the cache; with the
  cache every step processes one new token per sequence (decode), without
  it the whole prefix is recomputed. Both give the same logits (tested).
* `speculative_generate` - self-speculative decoding with the MTP module as
  the draft model (Sections 7.5 and 15.5): MTP proposes one extra token, the
  main model verifies it in the same forward pass. With ``temperature=0`` the
  output equals greedy decoding token for token; with sampling the
  acceptance rule min(1, p/q) plus the residual max(0, p - q) keeps the
  output distribution exactly that of the main model (formulas (7.7)-(7.9)).
* `quantize_int8` - weight-only symmetric INT8 with one scale per output
  channel (formula (15.4)).

CLI::

    python -m tinyseek.generate --run runs/v3 --prompt "Q: 12+34=?\\nA: "
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TinySeekModel


# ------------------------------------------------------------------ sampling
def filter_logits(logits: torch.Tensor, top_k: Optional[int] = None,
                  top_p: Optional[float] = None) -> torch.Tensor:
    """Mask logits outside the top-k set and outside the top-p nucleus."""
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p          # keep the smallest prefix with mass >= p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: Optional[int] = None, top_p: Optional[float] = None
                      ) -> torch.Tensor:
    """Sampling distribution p = softmax(filter(logits) / tau)."""
    if temperature <= 0:
        p = torch.zeros_like(logits, dtype=torch.float32)
        return p.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    return (filter_logits(logits.float(), top_k, top_p) / temperature).softmax(-1)


def pick(logits: torch.Tensor, temperature: float, top_k=None, top_p=None,
         generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """One token per row of ``logits`` (B, V)."""
    if temperature <= 0:
        return logits.argmax(-1)
    p = probs_from_logits(logits, temperature, top_k, top_p)
    return torch.multinomial(p, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_ids(model: TinySeekModel, ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, generator: Optional[torch.Generator] = None,
                 use_cache: bool = True, eos_id: Optional[int] = None,
                 return_logits: bool = False):
    """Generate continuations for a batch of equal-length prompts ``ids`` (B, T).

    Returns the new tokens (B, n) and, if ``return_logits``, the list of
    logits used at every step. Generation stops early once every row has
    produced ``eos_id``.
    """
    model.eval()
    ctx = model.config.max_position_embeddings
    cache = model.new_cache() if use_cache else None
    seq = ids
    out, all_logits = [], []
    done = torch.zeros(ids.shape[0], dtype=torch.bool)
    logits, _ = model(seq, cache=cache)
    for _ in range(max_new_tokens):
        last = logits[:, -1]
        if return_logits:
            all_logits.append(last)
        nxt = pick(last, temperature, top_k, top_p, generator)
        out.append(nxt)
        if eos_id is not None:
            done |= nxt == eos_id
            if bool(done.all()):
                break
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        if seq.shape[1] >= ctx:
            break
        if use_cache:
            logits, _ = model(nxt[:, None], cache=cache)
        else:
            logits, _ = model(seq)
    new = torch.stack(out, dim=1) if out else ids[:, :0]
    return (new, all_logits) if return_logits else new


def sample(model: TinySeekModel, tok, prompt: str, max_new_tokens: int = 64,
           temperature: float = 1.0, top_k: Optional[int] = None,
           top_p: Optional[float] = None, seed: Optional[int] = None,
           stop_at_eos: bool = True, use_cache: bool = True) -> str:
    """Continue ``prompt`` and return the generated text (Chapter 2)."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = torch.tensor([tok.encode(prompt)])
    new = generate_ids(model, ids, max_new_tokens, temperature, top_k, top_p, gen,
                       use_cache, tok.eos_id if stop_at_eos else None)[0].tolist()
    if stop_at_eos and tok.eos_id in new:
        new = new[:new.index(tok.eos_id)]
    return tok.decode(new)


# ---------------------------------------------------------- speculative decoding
def _truncate_cache(cache: dict, length: int) -> None:
    """Drop cached positions >= length (after a rejected draft token)."""
    for layer in cache["layers"]:
        for k, v in layer.items():
            layer[k] = v[:, :, :length] if v.dim() == 4 else v[:, :length]
    cache["len"] = length


def speculative_accept(p: torch.Tensor, q: torch.Tensor, draft: int,
                       generator: Optional[torch.Generator] = None) -> tuple[bool, int]:
    """Accept ``draft`` ~ q with probability min(1, p/q) (formula (7.7));
    otherwise resample from norm(max(0, p - q)) (7.8). Returns (accepted, token)."""
    ratio = float(p[draft] / q[draft].clamp_min(1e-20))
    u = float(torch.rand((), generator=generator))
    if u < min(1.0, ratio):
        return True, draft
    residual = (p - q).clamp_min(0)
    if float(residual.sum()) <= 0:
        residual = p
    return False, int(torch.multinomial(residual / residual.sum(), 1, generator=generator))


@torch.no_grad()
def speculative_generate(model: TinySeekModel, ids: list[int] | torch.Tensor,
                         max_new_tokens: int, temperature: float = 0.0,
                         top_k: Optional[int] = None, top_p: Optional[float] = None,
                         generator: Optional[torch.Generator] = None,
                         eos_id: Optional[int] = None) -> tuple[list[int], dict]:
    """Self-speculative decoding with the depth-1 MTP module as the draft.

    One iteration: MTP drafts token x_{n+1} from (h_{n-1}, x_n); the main model
    processes [x_n, draft] in one forward pass, which yields p(x_{n+1}) and
    p(x_{n+2}). If the draft is accepted the iteration emits two tokens
    (the draft and a bonus token), otherwise one corrected token.

    Returns the new tokens and statistics: ``acceptance`` (share of accepted
    drafts), ``tokens_per_step`` (tokens per main-model forward pass) and
    ``main_steps``.
    """
    if not len(model.mtp):
        raise ValueError("speculative_generate needs a model with an MTP module")
    model.eval()
    prompt = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    ctx = model.config.max_position_embeddings
    cache = model.new_cache()
    X = list(prompt)
    h = model.hidden_states(torch.tensor([X]), cache)                  # prefill
    H = [h[0, i] for i in range(h.shape[1])]                           # h_i of final tokens
    p_next = probs_from_logits(model.logits_from_hidden(h[:, -1])[0], temperature, top_k, top_p)
    X.append(_draw(p_next, temperature, generator))                    # pending x_n
    mtp_fed, accepted, drafted, steps = 0, 0, 0, 1
    while len(X) - len(prompt) < max_new_tokens and len(X) + 1 < ctx:
        if eos_id is not None and X[-1] == eos_id:
            break
        # 1) draft with MTP over the new final positions mtp_fed .. len(X)-2
        hs = torch.stack(H[mtp_fed:len(X) - 1])[None]
        nxt = torch.tensor([X[mtp_fed + 1:len(X)]])
        lg, _ = model.mtp_logits(0, hs, nxt, start=mtp_fed, cache=cache)
        mtp_fed = len(X) - 1
        q = probs_from_logits(lg[0, -1], temperature, top_k, top_p)
        d = _draw(q, temperature, generator)
        # 2) verify [x_n, draft] with the main model
        n_cached = cache["len"]
        hv = model.hidden_states(torch.tensor([[X[-1], d]]), cache)
        steps += 1
        drafted += 1
        pv = probs_from_logits(model.logits_from_hidden(hv)[0], temperature, top_k, top_p)
        if temperature <= 0:
            ok, tok = (int(pv[0].argmax()) == d), int(pv[0].argmax())
        else:
            ok, tok = speculative_accept(pv[0], q, d, generator)
        if ok:
            accepted += 1
            H += [hv[0, 0], hv[0, 1]]
            X.append(d)
            if eos_id is not None and d == eos_id:
                break
            X.append(_draw(pv[1], temperature, generator))             # bonus token
        else:
            _truncate_cache(cache, n_cached + 1)                       # forget the draft
            H.append(hv[0, 0])
            X.append(tok)
    new = X[len(prompt):len(prompt) + max_new_tokens]
    if eos_id is not None and eos_id in new:
        new = new[:new.index(eos_id) + 1]
    stats = {"acceptance": accepted / max(1, drafted), "main_steps": steps,
             "tokens_per_step": len(new) / max(1, steps), "drafted": drafted,
             "accepted": accepted}
    return new, stats


def _draw(p: torch.Tensor, temperature: float, generator) -> int:
    if temperature <= 0:
        return int(p.argmax())
    return int(torch.multinomial(p, 1, generator=generator))


# ---------------------------------------------------------------------- INT8
class Int8Linear(nn.Module):
    """Weight-only INT8 linear layer: w ~ s * q, q in [-127, 127], one scale per
    output channel (formula (15.4)); activations stay in floating point."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        self.register_buffer("qweight", torch.clamp(torch.round(w / scale), -127, 127)
                             .to(torch.int8))
        self.register_buffer("scale", scale)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight (MLA absorption reads ``wkv_b.weight``)."""
        return self.qweight.float() * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def quantize_int8(model: nn.Module, skip: tuple[str, ...] = ("gate",)) -> nn.Module:
    """Replace every ``nn.Linear`` (except names containing ``skip``) with
    `Int8Linear` in place; returns the model."""
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(mod, child_name, Int8Linear(child))
    return model


def weight_bytes(model: nn.Module) -> int:
    """Bytes of all parameters and buffers (INT8 weights count one byte)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total


def main(argv=None) -> None:
    from .train import load_checkpoint
    p = argparse.ArgumentParser(description="Generate text with a trained TinySeek "
                                "model (Chapters 2, 7, 15).")
    p.add_argument("--run", default="runs/v3", help="checkpoint directory")
    p.add_argument("--prompt", default="Q: 12+34=?\nA: ")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--speculative", action="store_true", help="MTP self-speculation")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    model, tok = load_checkpoint(args.run)
    prompt = args.prompt.encode().decode("unicode_escape") if "\\n" in args.prompt else args.prompt
    gen = torch.Generator().manual_seed(args.seed)
    t = time.time()
    if args.speculative:
        new, st = speculative_generate(model, tok.encode(prompt), args.max_new_tokens,
                                       args.temperature, args.top_k, args.top_p, gen,
                                       tok.eos_id)
        print(prompt + tok.decode(new, skip_special=True))
        print(f"acceptance {st['acceptance']:.2f}, tokens/step {st['tokens_per_step']:.2f}")
    else:
        print(prompt + sample(model, tok, prompt, args.max_new_tokens, args.temperature,
                              args.top_k, args.top_p, args.seed,
                              use_cache=not args.no_cache))
    print(f"[{time.time() - t:.2f}s]")


if __name__ == "__main__":
    main()
