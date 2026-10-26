"""Batched looped-forward engine for Ouro (right-padding layout).

Why this exists
---------------
Decoding a looped model is **weight-bandwidth bound**: every generated token
re-reads the full 1.4 B-parameter weight set once per loop, i.e. 4 x 2.87 GB =
~11.5 GB / token on a laptop GPU measured at ~173 GB/s.  Profiling gave
5.24 ms per decoder layer (attention 2.99 + MLP 0.97 + norms) and only ~7 ms
per token of launch overhead -- so kernel micro-optimisation cannot help, but
**batching can**: with ``B`` samples the weight reads are shared, giving close
to a ``B``-fold throughput gain.  That is what makes GSM8K / MATH-500 (which
need ~150-250 generated tokens per problem) feasible at all.

Layout
------
* ``input_ids`` is ``[B, S]``, right-padded; ``lens`` (``[B]``) holds the true
  prompt length of each sample.
* A unit's KV bank is ``[B, H, L_u, Dk]`` where ``L_u`` is the largest kept
  length in the batch; ``bank_lens[u]`` (``[B]``) records each sample's valid
  length so padded slots can be masked when decoding.
* Prefill reuses :func:`loopkv.attention_with_bank` (verified numerically
  comparable with the stock model); decoding uses
  :func:`loopkv.attention_decode`, which appends in place into a pre-allocated
  buffer.

Numerical note: importance scores and budgets are computed **per sample**
(they depend on that sample's own length), while all matmuls are batched.  The
computation is therefore identical to the unbatched engine, merely grouped.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from .loopkv import attention_decode, attention_with_bank, apply_rotary
from .policy import (DynaLoopConfig, allocate_anchor_budgets,
                     allocate_block_budgets, allocate_fixed_budgets,
                     allocate_js_budgets,
                     allocate_overlap_budgets,
                     js_divergence,
                     keep_indices_for_unit, window_query_scores)


def _padded_mask(seq_len: int, lens: torch.Tensor, dtype, device) -> torch.Tensor:
    """Additive mask ``[B, 1, S, S]``: causal AND padding-aware.

    Key ``k`` is hidden from query ``q`` when ``k > q`` (causal) or when
    ``k >= lens[b]`` (right-padding).
    """
    q = torch.arange(seq_len, device=device)[None, None, :, None]
    k = torch.arange(seq_len, device=device)[None, None, None, :]
    m = ((k > q) | (k >= lens.to(device)[:, None, None, None])).to(dtype)
    return m * torch.finfo(dtype).min


def _decode_pad_mask(lens: torch.Tensor, length: int, dtype, device) -> torch.Tensor:
    """bool ``[B, 1, 1, L]``: True on padded KV slots (to be masked)."""
    ar = torch.arange(length, device=device)[None, None, None, :]
    return ar >= lens.to(device)[:, None, None, None]


class BatchedOuroEngine:
    """Runs ``B`` prompts through the H-loop model sharing one weight read."""

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
        self._reset()

    def _reset(self):
        self.bank: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.bank_lens: Dict[Tuple[int, int], torch.Tensor] = {}
        self._kbuf: Dict[Tuple[int, int], torch.Tensor] = {}
        self._vbuf: Dict[Tuple[int, int], torch.Tensor] = {}
        self._fill: Dict[Tuple[int, int], torch.Tensor] = {}
        self.stats: Dict[str, float] = {}
        self._prompt_kv_tokens = 0
        self._prompt_kv_bytes = 0
        self._prompt_fill: Dict[Tuple[int, int], torch.Tensor] = {}
        # loop-anchored allocator: layer -> {sample_index: reference-loop score}
        # "prev"  -> overwritten after every loop, so loop t sees loop t-1
        # "loop1" -> written once, when loop 1 is produced
        self._ref: Dict[int, Dict[int, torch.Tensor]] = {}
        # "rel" js allocator: per-sample calibration state (D1 median list)
        self._js_stats: Dict[int, Dict] = {}
        # (heads, head_dim, bytes_per_token_for_K_and_V) per unit
        self._unit_meta: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    # ------------------------------------------------------------------ #
    # scoring (per sample, because lengths differ)                         #
    # ------------------------------------------------------------------ #

    def _scores_for_unit(self, attn_mod, hidden_window: torch.Tensor,
                         k_full: torch.Tensor, seq_len: int,
                         cos: torch.Tensor, sin: torch.Tensor,
                         q_pos: torch.Tensor) -> torch.Tensor:
        """Importance of every key position of ONE sample: ``[S]``."""
        q = attn_mod.q_proj(hidden_window)
        q = q.view(hidden_window.shape[0], hidden_window.shape[1], -1,
                   attn_mod.head_dim).transpose(1, 2)
        k = attn_mod.k_proj(hidden_window)
        k = k.view(hidden_window.shape[0], hidden_window.shape[1], -1,
                   attn_mod.head_dim).transpose(1, 2)
        # keys come from the bank (already RoPE'd); only queries need rotating
        q, _ = apply_rotary(q, k, cos, sin)
        return window_query_scores(q, k_full, attn_mod.scaling, causal=True,
                                   q_positions=q_pos)[0]

    # ------------------------------------------------------------------ #
    # prefill                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """``input_ids``: ``[B, S]`` right-padded; ``lens``: ``[B]``."""
        self._reset()
        model = self.model.model
        B, S = input_ids.shape
        hidden = model.embed_tokens(input_ids)
        device = hidden.device
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pe = model.rotary_emb(hidden, position_ids)
        self._pe = pe
        ws = min(self.cfg.window_size, S)
        self._ws = ws
        mask = _padded_mask(S, lens, hidden.dtype, device)

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
                        attn, x, None, None, pe, mask)
                    kvs[(u_ut, layer_idx)] = (k_new, v_new)
                    # per-sample observation window: each sample's own last
                    # ``ws`` *real* tokens, never the padding region
                    win_hidden[(u_ut, layer_idx)] = [
                        x[b:b + 1, max(0, int(lens[b]) - ws):int(lens[b]), :]
                        .detach().clone() for b in range(B)]
                    x = layer.input_layernorm_2(out)
                    hidden = hidden + x
                    res = hidden
                    h = layer.post_attention_layernorm(hidden)
                    h = layer.mlp(h)
                    h = layer.post_attention_layernorm_2(h)
                    hidden = res + h
                self._compress_block(block_units, win_hidden, kvs, S, lens,
                                     block_no=start // block_size)
            hidden = model.norm(hidden)
        self.stats["prefill_cost"] = time.time() - t0
        # snapshot of the *prompt* KV length per unit per sample
        self._prompt_fill = ({u: t.clone() for u, t in self._fill.items()}
                             if self._fill else
                             {u: self.bank_lens[u].clone() for u in self.bank})
        self._prompt_loop_ret = self._compute_loop_retention(lens)

        # each sample's own last real position
        idx = (lens.to(device) - 1).view(B, 1, 1).expand(-1, 1, hidden.shape[-1])
        last = hidden.gather(1, idx)
        self._prompt_kv_tokens = self.kv_tokens()
        self._prompt_kv_bytes = self.kv_bytes()
        return self.model.lm_head(last)

    def _compress_block(self, block_units, win_hidden, kvs, seq_len, lens,
                        block_no):
        """Per-sample budget + ChunkKV block selection, then pad into one bank.

        The budget allocation is done **for the whole block at once**, exactly
        as in the unbatched engine: DynamicKV's progressive update needs to see
        all ``m`` units of the block to redistribute budget between them, so
        calling it once per unit would silently change the policy.
        """
        B = len(lens)
        if not self.cfg.enabled:
            for u in block_units:
                self.bank[u] = kvs[u]
                self.bank_lens[u] = lens.to(self.device)
                kk, vv = kvs[u]
                self._unit_meta[u] = (kk.shape[1], kk.shape[3],
                                      kk.element_size() + vv.element_size())
            return

        keeps: Dict[Tuple[int, int], List[torch.Tensor]] = {u: []
                                                            for u in block_units}
        src = {u: self.model.model.layers[u[1]].self_attn for u in block_units}

        for b in range(B):
            n_b = int(lens[b].item())
            s = max(0, n_b - self._ws)
            cos = self._pe[0][b:b + 1, s:n_b, :]
            sin = self._pe[1][b:b + 1, s:n_b, :]
            q_pos = torch.arange(s, n_b, device=self.device)
            scores_b = {}
            for u in block_units:
                scores_b[u] = self._scores_for_unit(
                    src[u], win_hidden[u][b],
                    kvs[u][0][b:b + 1, :, :n_b, :], n_b, cos, sin, q_pos)

            if self.cfg.allocator == "fixed":
                budgets = allocate_fixed_budgets(block_units, scores_b, n_b,
                                                 self.cfg)
            elif self.cfg.allocator == "js":
                # anchor = this layer's scores in the PREVIOUS loop; the js
                # allocator turns the displacement into this unit's budget.
                refs_b = {l: v[b] for l, v in self._ref.items() if b in v}
                budgets = allocate_js_budgets(block_units, scores_b, n_b,
                                             self.cfg, refs_b,
                                             js_stats=self._js_stats.setdefault(b, {}))
                if (self.cfg.anchor_source == "prev"
                        or block_units[0][0] == 0):
                    for (u_ut, u_l) in block_units:
                        if u_ut == block_units[0][0]:
                            self._ref.setdefault(u_l, {})[b] = \
                                scores_b[(u_ut, u_l)].detach().clone()
            elif self.cfg.allocator in ("anchor", "overlap"):
                # Anchor bookkeeping *before* budgeting: ``_ref`` currently
                # holds the reference loop's scores (loop 1 on the very first
                # block, otherwise loop t-1 -- see the update at the end).
                refs_b = {l: v[b] for l, v in self._ref.items() if b in v}
                fn = (allocate_anchor_budgets if self.cfg.allocator == "anchor"
                      else allocate_overlap_budgets)
                budgets = fn(block_units, scores_b, n_b, self.cfg, refs_b)
                # refresh the anchor with THIS loop's scores, so that the
                # next loop is measured against it ("prev") or so that loop 1
                # stays frozen as the permanent anchor ("loop1").
                if (self.cfg.anchor_source == "prev"
                        or block_units[0][0] == 0):
                    for (u_ut, u_l) in block_units:
                        if u_ut == block_units[0][0]:
                            self._ref.setdefault(u_l, {})[b] = \
                                scores_b[(u_ut, u_l)].detach().clone()
            else:
                budgets = allocate_block_budgets(
                    block_units, scores_b, n_b, self.cfg,
                    n_processed=(block_no + 1) * len(block_units))
            for u in block_units:
                if budgets[u] >= n_b:
                    keep = torch.arange(n_b, device=self.device)
                else:
                    keep = keep_indices_for_unit(
                        scores_b[u], budgets[u], n_b, self.cfg.window_size,
                        self.cfg.sink_tokens,
                        chunk_size=self.cfg.chunk_size)
                keeps[u].append(keep)

        for u in block_units:
            lens_u = [int(k.numel()) for k in keeps[u]]
            max_n = max(lens_u)
            Hh, Dk = kvs[u][0].shape[1], kvs[u][0].shape[3]
            kb = torch.zeros((B, Hh, max_n, Dk), dtype=kvs[u][0].dtype,
                             device=self.device)
            vb = torch.zeros_like(kb)
            for b in range(B):
                kb[b:b + 1, :, :lens_u[b], :].copy_(
                    kvs[u][0][b:b + 1].index_select(2, keeps[u][b]))
                vb[b:b + 1, :, :lens_u[b], :].copy_(
                    kvs[u][1][b:b + 1].index_select(2, keeps[u][b]))
            self.bank[u] = (kb, vb)
            self.bank_lens[u] = torch.tensor(lens_u, device=self.device)
            self._unit_meta[u] = (kb.shape[1], kb.shape[3],
                                  kb.element_size() + vb.element_size())

    # ------------------------------------------------------------------ #
    # decode                                                               #
    # ------------------------------------------------------------------ #

    def prepare_decode(self, max_new_tokens: int):
        """Copy the prompt KV into growable buffers and **release the bank**.

        The copy is done unit by unit and each source tensor is dropped right
        after it has been consumed.  Allocating all buffers up front would hold
        the prompt KV twice (once in ``bank``, once in ``_kbuf``); on an 8 GB
        card that pushed batch size 8 from ~0.8 s/step to 5.5 s/step because the
        allocator started thrashing.
        """
        self._kbuf, self._vbuf, self._fill = {}, {}, {}
        for u in list(self.bank.keys()):
            k, v = self.bank.pop(u)
            B, Hh, n0, Dk = k.shape
            cap = n0 + int(max_new_tokens)
            kb = torch.empty((B, Hh, cap, Dk), dtype=k.dtype, device=k.device)
            vb = torch.empty((B, Hh, cap, Dk), dtype=v.dtype, device=v.device)
            kb[:, :, :n0, :].copy_(k)
            vb[:, :, :n0, :].copy_(v)
            # CRITICAL: zero the growable tail.
            # ``torch.empty`` leaves the tail (n0:cap) uninitialised, and on
            # this allocator those bytes frequently decode as NaN.  Decode
            # reads ``k_view = buf[:, :, :L, :]`` with ``L = max(fill)``, so a
            # short sample sees slots beyond *its own* fill -- padded or
            # uninitialised.  The pad mask zeroes their attention probability,
            # but ``0 * NaN = NaN`` still propagates through the value matmul
            # (the K side is safe because ``masked_fill`` overwrites those
            # logits before the softmax).  Symptom: the padded samples emit a
            # garbage/EOS token after one step, i.e. silently truncated
            # generations.  Zeroing once here costs a single memset per unit.
            if cap > n0:
                kb[:, :, n0:, :].zero_()
                vb[:, :, n0:, :].zero_()
            self._kbuf[u], self._vbuf[u] = kb, vb
            self._fill[u] = self.bank_lens[u].clone()
            del k, v

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor,
                    positions: torch.Tensor) -> torch.Tensor:
        """``token_ids``: ``[B, 1]``; ``positions``: ``[B]`` (per-sample)."""
        model = self.model.model
        hidden = model.embed_tokens(token_ids)
        pe = model.rotary_emb(hidden, positions.view(-1, 1))
        for ut in range(self.H):
            for layer_idx, layer in enumerate(model.layers):
                u = (ut, layer_idx)
                attn = layer.self_attn
                f = self._fill[u]
                L = int(f.max().item())
                k_view = self._kbuf[u][:, :, :L, :]
                v_view = self._vbuf[u][:, :, :L, :]
                pad = _decode_pad_mask(f, L, hidden.dtype, hidden.device)
                x = layer.input_layernorm(hidden)
                out, k_new, v_new = attention_decode(
                    attn, x, k_view, v_view, pe, pad)
                # write each sample's new K/V at its own position
                b_idx = torch.arange(k_new.shape[0], device=k_new.device)
                self._kbuf[u][b_idx, :, f, :] = k_new[:, :, 0, :]
                self._vbuf[u][b_idx, :, f, :] = v_new[:, :, 0, :]
                self._fill[u] = f + 1
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
    # generation                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, lens: torch.Tensor,
                 max_new_tokens: int = 64, temperature: float = 0.0,
                 stop_ids: Optional[List[int]] = None
                 ) -> Tuple[List[List[int]], List[int]]:
        """Greedy batched generation.

        Returns ``(sequences, valid_lengths)``.  ``valid_lengths[b]`` is where
        sample ``b`` first hit EOS / a stop id, so callers decode only that
        prefix; finished samples keep advancing (cheap) but their output is
        ignored.
        """
        B = input_ids.shape[0]
        logits = self.prefill(input_ids, lens)
        self.prepare_decode(max_new_tokens)
        pos = lens.to(self.device).clone()
        out: List[List[int]] = [[] for _ in range(B)]
        valid = [max_new_tokens for _ in range(B)]
        stop = set(stop_ids or [])
        eos = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        if eos is not None:
            stop.add(int(eos))

        t0 = time.time()
        for step in range(max_new_tokens):
            if temperature == 0:
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids_list = nxt.view(-1).tolist()
            for b in range(B):
                out[b].append(ids_list[b])
                if valid[b] == max_new_tokens and ids_list[b] in stop:
                    valid[b] = step + 1
            if step + 1 < max_new_tokens:
                logits = self.decode_step(nxt, pos)
                pos = pos + 1
        self.stats["decode_cost"] = time.time() - t0
        self.stats["generated"] = max_new_tokens
        return out, valid

    # ------------------------------------------------------------------ #
    # accounting                                                           #
    # ------------------------------------------------------------------ #

    def _fill_snapshot(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Per-unit, per-sample counts of *prompt* KV tokens (not padding)."""
        return self._prompt_fill if self._prompt_fill else self._fill

    def _compute_loop_retention(self, lens) -> dict:
        """Realised mean KV retention per loop over the last prefill.

        ``{loop1: 1.0, loop2: 0.57, ...}`` -- mean over layers of
        (kept prompt KV tokens) / (prompt length), averaged per loop.
        """
        try:
            out = {}
            L = lens.to(self.device).float().clamp_min(1.0)
            for ut in range(self.H):
                vals = []
                for l in range(self.M):
                    f = self._prompt_fill.get((ut, l))
                    if f is None:
                        continue
                    vals.append(float((f.to(self.device).float() / L).mean()))
                if vals:
                    out[f"loop{ut + 1}"] = round(min(sum(vals) / len(vals), 1.0), 4)
            return out
        except Exception:
            return {}

    def per_loop_retention(self, lens: torch.Tensor) -> Dict[str, float]:
        """Public wrapper: realised mean retention of each loop.

        Returns ``{"loop1": 1.0, "loop2": 0.53, "loop3": 0.15, ...}``.
        """
        return self._compute_loop_retention(lens)

    def per_unit_retention(self, lens: torch.Tensor) -> Dict[tuple, float]:
        """Realised retention of every ``(loop, layer)`` unit.

        Handy for the per-(loop, layer) heatmap in ``examples/``.
        """
        out: Dict[tuple, float] = {}
        L = lens.to(self.device).float().clamp_min(1.0)
        for u, f in self._prompt_fill.items():
            out[u] = float((f.to(self.device).float() / L).mean())
        return out

    def kv_tokens(self) -> int:
        """Prompt KV tokens kept, summed over units and samples."""
        src = self._fill_snapshot()
        if src:
            return int(sum(int(t.sum().item()) for t in src.values()))
        return int(sum(k.shape[2] for k, _ in self.bank.values()))

    def kv_bytes(self) -> int:
        """Prompt KV bytes (bf16, K and V), padding excluded."""
        src = self._fill_snapshot()
        if src:
            total = 0
            for u, t in src.items():
                heads, hdim, esz = self._unit_meta[u]
                total += int(t.sum().item()) * heads * hdim * esz
            return int(total)
        return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                       for k, v in self.bank.values()))

    def retention_ratio(self, lens: torch.Tensor) -> float:
        full = float(sum(int(x) for x in lens)) * self.H * self.M
        return self.kv_tokens() / full if full else 0.0

    def per_sample_retention(self, lens: torch.Tensor) -> List[float]:
        """Retention ratio of each sample in the batch."""
        B = len(lens)
        kept = torch.zeros(B, device=lens.device, dtype=torch.float32)
        for t in self._prompt_fill.values():
            kept += t.float()
        return (kept / (lens.float() * self.H * self.M)).tolist()
