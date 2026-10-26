"""Minimal Ouro looped-forward engine with per-(loop, layer) KV banks.

Mirrors ``OuroModel.forward`` / ``OuroDecoderLayer.forward`` exactly
(embed -> [loop{ layers -> norm }] -> lm_head) but keeps the KV of every
``(ut, layer)`` unit under our own control, so a budget policy can evict
tokens right after the unit is written.

Two modes:
* ``policy.enabled=False``  -> Full KV baseline (B0).
* ``policy.enabled=True``   -> DynaLoop: first ``K`` loops full, rest
  compressed with DynamicKV-style adaptive budgets.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from .loopkv import (attention_decode, attention_with_bank, apply_rotary,
                    build_causal_mask)
from .policy import (DynaLoopConfig, allocate_block_budgets,
                     keep_indices_for_unit, window_query_scores)


def _n_runs(sorted_idx: torch.Tensor) -> int:
    """Number of contiguous runs in a sorted index tensor.

    1 run = the kept KV is one unbroken block; N runs = N scattered fragments.
    This is the direct measure of "block granularity" for the report.
    """
    if sorted_idx.numel() == 0:
        return 0
    d = (sorted_idx[1:] - sorted_idx[:-1]) != 1
    return int(d.sum().item()) + 1


class OuroLoopEngine:
    def __init__(self, model, tokenizer, policy: Optional[DynaLoopConfig] = None,
                 device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = policy if policy is not None else DynaLoopConfig(enabled=False)
        self.device = device
        self.M = int(model.config.num_hidden_layers)
        self.H = int(getattr(model.config, "total_ut_steps", 4))
        self.cfg.num_loops = self.H
        self.cfg.num_layers = self.M
        self.bank: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #

    def _reset(self):
        self.bank = {}
        self.stats = {}
        self._chunk_stats: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        self._kbuf: Dict[Tuple[int, int], torch.Tensor] = {}
        self._vbuf: Dict[Tuple[int, int], torch.Tensor] = {}
        self._fill: Dict[Tuple[int, int], int] = {}
        # Post-prefill accounting: the compression ratio must be measured on
        # the *prompt* KV, not on the prompt+generated cache -- otherwise a
        # short prompt with many generated tokens reports retention > 1.
        self._prompt_kv_tokens = 0
        self._prompt_kv_bytes = 0

    # ------------------------------------------------------------------ #
    # decode buffers                                                      #
    # ------------------------------------------------------------------ #

    def prepare_decode(self, max_new_tokens: int):
        """Pre-allocate one K/V buffer per (loop, layer) unit.

        Prefill leaves ``n_pre`` tokens in each unit's bank; decoding appends
        ``max_new_tokens`` more. Allocating ``n_pre + max_new_tokens`` up front
        removes the per-token ``cat`` that used to copy every bank (H*M times
        per generated token).
        """
        self._kbuf, self._vbuf, self._fill = {}, {}, {}
        for u, (k, v) in self.bank.items():
            n0 = int(k.shape[2])
            cap = n0 + int(max_new_tokens)
            kb = torch.empty((k.shape[0], k.shape[1], cap, k.shape[3]),
                             dtype=k.dtype, device=k.device)
            vb = torch.empty((v.shape[0], v.shape[1], cap, v.shape[3]),
                             dtype=v.dtype, device=v.device)
            kb[:, :, :n0].copy_(k)
            vb[:, :, :n0].copy_(v)
            self._kbuf[u], self._vbuf[u], self._fill[u] = kb, vb, n0

    def chunk_summary(self) -> dict:
        """Aggregate (budget, kept, #contiguous-runs) over the compressed units."""
        budgets, kept, runs = [], [], []
        for _u, recs in self._chunk_stats.items():
            for b, k, r in recs:
                budgets.append(b)
                kept.append(k)
                runs.append(r)
        if not budgets:
            return {}
        n = len(budgets)
        return {
            "units": n,
            "budget_tokens_mean": round(sum(budgets) / n, 1),
            "kept_tokens_mean": round(sum(kept) / n, 1),
            "contiguous_runs_mean": round(sum(runs) / n, 2),
            "longest_run_mean": round(sum(kept) / n / max(sum(runs) / n, 1e-9), 2),
        }

    def _scores_for_unit(self, attn_mod, hidden_window: torch.Tensor,
                         k_full: torch.Tensor, seq_len: int, ws: int) -> torch.Tensor:
        """Importance of every key position: attention of the window queries."""
        # hidden_window: [1, ws, D] (already layernormed) -> q states
        q = attn_mod.q_proj(hidden_window)
        q = q.view(hidden_window.shape[0], hidden_window.shape[1], -1,
                   attn_mod.head_dim).transpose(1, 2)
        k = attn_mod.k_proj(hidden_window)
        k = k.view(hidden_window.shape[0], hidden_window.shape[1], -1,
                   attn_mod.head_dim).transpose(1, 2)
        # RoPE is only needed for the query side here (keys come from the bank)
        cos, sin = self._pe_window
        q, _ = apply_rotary(q, k, cos, sin)
        # ``q``/``k_full`` are 4-D [1, H, w, Dk] / [1, H, S, Dk]
        return window_query_scores(q, k_full, attn_mod.scaling, causal=True)[0]

    def _compress_block(self, block_units, win_hidden, kvs, seq_len, block_no):
        """DynamicKV progressive update: score + budget + evict, per block."""
        if not self.cfg.enabled:
            for u in block_units:
                self.bank[u] = kvs[u]
            return

        scores = {}
        for u in block_units:
            ut, layer_idx = u
            attn = self.model.model.layers[layer_idx].self_attn
            k_full = kvs[u][0]
            scores[u] = self._scores_for_unit(attn, win_hidden[u], k_full,
                                              seq_len, self.cfg.window_size)
        budgets = allocate_block_budgets(block_units, scores, seq_len,
                                         self.cfg, n_processed=(block_no + 1) * len(block_units))
        for u in block_units:
            b = budgets[u]
            if b >= seq_len:
                self.bank[u] = kvs[u]
                continue
            keep = keep_indices_for_unit(scores[u], b, seq_len,
                                         self.cfg.window_size, self.cfg.sink_tokens,
                                         chunk_size=self.cfg.chunk_size)
            k, v = kvs[u]
            self.bank[u] = (k.index_select(2, keep), v.index_select(2, keep))
            self._chunk_stats.setdefault(u, []).append(
                (int(b), int(keep.numel()), _n_runs(keep)))

    # ------------------------------------------------------------------ #
    # prefill                                                             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the full H-loop prefill; returns logits of the last position."""
        self._reset()
        model = self.model.model
        hidden = model.embed_tokens(input_ids)
        bsz, seq_len, _ = hidden.shape
        device = hidden.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pe = model.rotary_emb(hidden, position_ids)
        # window slice of the position embeddings, reused for scoring
        ws = min(self.cfg.window_size, seq_len)
        self._pe_window = (pe[0][:, -ws:, :], pe[1][:, -ws:, :])
        causal = build_causal_mask(seq_len, device, hidden.dtype)

        block_size = max(int(self.cfg.block_size), 1)
        t0 = time.time()
        for ut in range(self.H):
            for start in range(0, self.M, block_size):
                block_units = [(ut, l) for l in range(start, min(start + block_size, self.M))]
                kvs, win_hidden = {}, {}
                for (u_ut, layer_idx) in block_units:
                    layer = model.layers[layer_idx]
                    attn = layer.self_attn
                    x = layer.input_layernorm(hidden)
                    out, k_new, v_new = attention_with_bank(
                        attn, x, None, None, pe, causal)
                    kvs[(u_ut, layer_idx)] = (k_new, v_new)
                    win_hidden[(u_ut, layer_idx)] = x[:, -ws:, :].detach().clone()
                    x = layer.input_layernorm_2(out)
                    hidden = hidden + x
                    res = hidden
                    h = layer.post_attention_layernorm(hidden)
                    h = layer.mlp(h)
                    h = layer.post_attention_layernorm_2(h)
                    hidden = res + h
                self._compress_block(block_units, win_hidden, kvs, seq_len,
                                     block_no=start // block_size)
            hidden = model.norm(hidden)
        self.stats["prefill_cost"] = time.time() - t0
        self.stats["prompt_tokens"] = seq_len
        self._prompt_kv_tokens = self.kv_tokens()
        self._prompt_kv_bytes = self.kv_bytes()
        self.stats["prompt_kv_tokens"] = self._prompt_kv_tokens
        self.stats["prompt_kv_MiB"] = round(self._prompt_kv_bytes / 1024 ** 2, 2)
        logits = self.model.lm_head(hidden[:, -1:, :])
        self._last_hidden = hidden[:, -1:, :]
        return logits

    # ------------------------------------------------------------------ #
    # decode                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, position: int) -> torch.Tensor:
        model = self.model.model
        hidden = model.embed_tokens(token_ids)
        position_ids = torch.tensor([[position]], device=hidden.device)
        pe = model.rotary_emb(hidden, position_ids)
        use_buf = bool(self._kbuf)
        for ut in range(self.H):
            for layer_idx, layer in enumerate(model.layers):
                u = (ut, layer_idx)
                attn = layer.self_attn
                x = layer.input_layernorm(hidden)
                if use_buf:
                    f = self._fill[u]
                    out, k_new, v_new = attention_decode(
                        attn, x, self._kbuf[u][:, :, :f, :],
                        self._vbuf[u][:, :, :f, :], pe)
                    # in-place append into the buffer; no reallocation
                    self._kbuf[u][:, :, f:f + 1, :].copy_(k_new)
                    self._vbuf[u][:, :, f:f + 1, :].copy_(v_new)
                    self._fill[u] = f + 1
                else:
                    # legacy path (no pre-allocated buffer): full-bank cat
                    k_bank, v_bank = self.bank[u]
                    out, k_new, v_new = attention_with_bank(
                        attn, x, k_bank, v_bank, pe, None)
                    self.bank[u] = (torch.cat([k_bank, k_new], dim=2),
                                    torch.cat([v_bank, v_new], dim=2))
                x = layer.input_layernorm_2(out)
                hidden = hidden + x
                res = hidden
                h = layer.post_attention_layernorm(hidden)
                h = layer.mlp(h)
                h = layer.post_attention_layernorm_2(h)
                hidden = res + h
            hidden = model.norm(hidden)
        return self.model.lm_head(hidden[:, -1:, :])

    # ------------------------------------------------------------------ #
    # generation                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.0, stop_at_newline: bool = False,
                 use_buffer: bool = True) -> torch.Tensor:
        seq = input_ids
        logits = self.prefill(seq)
        if use_buffer:
            self.prepare_decode(max_new_tokens)
        generated: List[int] = []
        pos = seq.shape[1]
        t0 = time.time()
        for _ in range(max_new_tokens):
            if temperature == 0:
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            tid = int(next_id.item())
            generated.append(tid)
            if self.tokenizer is not None and tid == self.tokenizer.eos_token_id:
                break
            if stop_at_newline and tid in self._newline_ids():
                break
            logits = self.decode_step(next_id, pos)
            pos += 1
        self.stats["decode_cost"] = time.time() - t0
        self.stats["generated"] = len(generated)
        return torch.tensor([generated], device=seq.device)

    def _newline_ids(self):
        if not hasattr(self, "_nl_ids") and self.tokenizer is not None:
            ids = set()
            for s in ("\n", "\n\n"):
                try:
                    ids.update(self.tokenizer.encode(s, add_special_tokens=False))
                except Exception:
                    pass
            self._nl_ids = ids
        return getattr(self, "_nl_ids", set())

    # ------------------------------------------------------------------ #
    # accounting                                                          #
    # ------------------------------------------------------------------ #

    def kv_tokens(self) -> int:
        """Total cached tokens summed over all (loop, layer) units.

        During decoding the live state lives in the pre-allocated buffers, whose
        occupied length is ``_fill``; ``bank`` still holds the post-prefill
        (compressed) KV and is used when no buffer has been allocated.
        """
        if self._fill:
            return int(sum(self._fill.values()))
        return int(sum(k.shape[2] for k, _ in self.bank.values()))

    def kv_bytes(self) -> int:
        """**Prompt** KV footprint (bf16), which is what the policy compresses.

        ``total_kv_bytes()`` additionally counts the tokens appended during
        decoding; those are identical for every configuration and would only
        dilute the measured compression ratio.
        """
        if self._prompt_kv_bytes:
            return int(self._prompt_kv_bytes)
        if self._fill:
            total = 0
            for u, n in self._fill.items():
                kb, vb = self._kbuf[u], self._vbuf[u]
                per = kb.shape[1] * kb.shape[3]      # heads * head_dim
                total += n * per * (kb.element_size() + vb.element_size())
            return int(total)
        return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                       for k, v in self.bank.values()))

    def total_kv_bytes(self) -> int:
        """Prompt **plus** generated KV (the real peak cache at the end)."""
        if self._fill:
            total = 0
            for u, n in self._fill.items():
                kb, vb = self._kbuf[u], self._vbuf[u]
                per = kb.shape[1] * kb.shape[3]
                total += n * per * (kb.element_size() + vb.element_size())
            return int(total)
        return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                       for k, v in self.bank.values()))

    def retention_ratio(self, seq_len: int) -> float:
        """Prompt-KV retention: kept prompt tokens / (H*M*S).

        Uses the post-prefill snapshot so it stays in [0, 1] regardless of how
        many tokens are generated afterwards.
        """
        full = self.H * self.M * seq_len
        kept = (self._prompt_kv_tokens if self._prompt_kv_tokens
                else self.kv_tokens())
        return kept / full if full else 0.0
