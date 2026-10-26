"""DynaLoop KV bank + manual attention for looped Transformers (Ouro).

Design notes
------------
Ouro's cache is indexed by ``ut * num_hidden_layers + layer_idx``
(``modeling_ouro.py:320``), i.e. **every (loop, layer) pair owns an
independent K/V tensor**.  With H loops and M layers there are H*M cache
units; a naive run therefore stores H full KV caches.

Two facts make "compress-on-write" valid and memory-cheap:

* unit (ut, l) is read **only** by layer ``l`` at loop ``ut`` -- once the
  layer returns, that unit is never touched again until decoding;
* consequently evicting tokens from unit (ut, l) right after it is written
  produces exactly the same decode-time cache as compressing everything
  after prefill, while the prefill peak memory is much lower.

All attention here re-implements ``OuroAttention.forward`` /
``eager_attention_forward`` (``modeling_ouro.py:228-344``) verbatim so the
baseline is bit-comparable with the stock model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def build_causal_mask(seq_len: int, device, dtype) -> torch.Tensor:
    """Additive causal mask ``[1, 1, S, S]`` (0 keep, -inf mask)."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def attention_with_bank(
    attn_mod,
    hidden_states: torch.Tensor,          # [B, S, D] already layernormed
    k_bank: torch.Tensor,                 # [B, H, Sk, Dk]
    v_bank: torch.Tensor,                 # [B, H, Sk, Dk]
    position_embeddings,                  # (cos, sin)
    causal_mask: Optional[torch.Tensor],  # additive [1,1,S,Sk] or None
    write_q_window: Optional[torch.Tensor] = None,  # reserved
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one attention step against an explicit KV bank.

    Returns ``(attn_output, k_new, v_new)`` where ``k_new/v_new`` are the
    freshly computed (RoPE-applied) K/V of this step, and ``attn_output`` is
    the projected output in ``[B, S, D]``.  The caller concatenates
    ``k_new/v_new`` into the bank itself, so the mask can be built for the
    exact key length.
    """
    bsz, q_len, _ = hidden_states.shape
    hidden_shape = (bsz, q_len, -1, attn_mod.head_dim)

    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary(query_states, key_states, cos, sin)

    k_full = torch.cat([k_bank, key_states], dim=2) if k_bank is not None else key_states
    v_full = torch.cat([v_bank, value_states], dim=2) if v_bank is not None else value_states

    attn_weights = torch.matmul(query_states, k_full.transpose(2, 3)) * attn_mod.scaling
    if causal_mask is not None:
        attn_weights = attn_weights + causal_mask[..., : k_full.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, v_full)

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def attention_decode(
    attn_mod,
    hidden_states: torch.Tensor,          # [B, 1, D] already layernormed
    k_view: torch.Tensor,                 # [B, H, Sk, Dk] *view* into the pre-allocated buffer
    v_view: torch.Tensor,                 # [B, H, Sk, Dk]
    position_embeddings,                  # (cos, sin)
    pad_mask: Optional[torch.Tensor] = None,  # bool [B, 1, 1, Sk] -> -inf
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-query decoding step against a pre-allocated KV buffer.

    Numerically identical to :func:`attention_with_bank` with ``causal_mask=None``
    and a freshly-catted bank, but it **never copies the bank**: the query is
    scored against the bank view and the new K/V separately, the two logit
    slices are concatenated (shape ``[B, H, 1, Sk+1]`` -- tiny), and the two
    value contributions are accumulated.

    This is the hot path: the old code did ``cat([bank, new])`` per unit, i.e.
    ``H*M`` full-bank copies (≈540 MB) per generated token.

    ``pad_mask`` hides the padding slots that appear when a batch mixes prompts
    of different lengths (right-padding layout).
    """
    bsz, q_len, _ = hidden_states.shape
    hidden_shape = (bsz, q_len, -1, attn_mod.head_dim)

    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary(query_states, key_states, cos, sin)

    # logits over [bank | new] without materialising the concatenated K/V
    logits_bank = torch.matmul(query_states, k_view.transpose(2, 3)) * attn_mod.scaling
    if pad_mask is not None:
        logits_bank = logits_bank.masked_fill(pad_mask, torch.finfo(logits_bank.dtype).min)
    logits_new = torch.matmul(query_states, key_states.transpose(2, 3)) * attn_mod.scaling
    logits = torch.cat([logits_bank, logits_new], dim=-1)

    probs = torch.softmax(logits.float(), dim=-1).to(value_states.dtype)
    sk = k_view.shape[-2]
    attn_output = (torch.matmul(probs[..., :sk], v_view)
                   + torch.matmul(probs[..., sk:], value_states))

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


def apply_rotary(q, k, cos, sin):
    """``apply_rotary_pos_emb`` from modeling_ouro.py (unsqueeze_dim=1)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
