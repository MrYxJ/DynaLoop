"""DynaLoop budget allocation: DynamicKV's adaptive budget, lifted to loops.

DynamicKV (Zhou et al., EMNLP 2025 Findings, §4) -- what we reuse verbatim:

* a **current window** of the last ``ws`` tokens is always kept;
* a per-layer maximum budget ``B_l = (w_t - w_s) * r_max``;
* importance = attention of the **window tokens** on the prompt keys,
  ``A_l = Pooling(A[:, ws])`` (multi-head pooling, here: sum over heads and
  over the window queries);
* a **progressive cache update**: every ``m`` layers the counts of the
  globally top-ranked entries are re-aggregated,
  ``C_l = Norm( (1/n) * Phi(TopK(A, (w_t-w_s) * n)) )``,
  ``Z_l = B_l * C_l / max(C)``, then renormalised so the block respects
  ``B = (w_t - w_s) * L``.

What is new here (the DynaLoop part):

* the budget unit is **(loop, layer)** instead of (layer) -- loops are split
  into ``keep_loops`` (full KV, ratio 1.0) and the remaining loops which
  share the rest of the global budget;
* ``min_budget_ratio`` (b_min): DynamicKV has **no** lower bound in the
  paper, so we add one and ablate it (see RECON.md R1.2 / §9 conflict #5);
* ``loop_decay``: optional monotone shrink of the retention ratio with loop
  depth (the looped analogue of PyramidKV's layer pyramid);
* ``chunk_size``: **block granularity** -- ChunkKV (arXiv 2502.00299, Alg. 1)
  groups *contiguous* tokens into chunks of length ``c``, scores a chunk by the
  **sum** of its token scores, and keeps whole chunks in the original sequence
  order. ``c = 1`` degenerates to the previous token-level top-k. Chunking is
  done on the **full** key sequence starting at index 0 (as in the paper), and
  ``k = floor(budget / c)`` chunks are kept, so the realised token count is
  ``k * c <= budget`` -- chunks are never split.

Inputs are per-unit score vectors ``s_u[j]`` (importance of key position j
for unit u); the caller builds them from the real attention of the window
queries, so no extra forward pass is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class DynaLoopConfig:
    num_loops: int = 4
    num_layers: int = 24
    keep_loops: int = 2            # first K loops keep full KV
    global_ratio: float = 0.55     # rho: target global KV retention ratio
    window_size: int = 64          # ws: always-kept recent tokens
    min_budget_ratio: float = 0.05  # b_min: per-unit lower bound (our addition)
    block_size: int = 6            # m: progressive-update block (layers)
    chunk_size: int = 1            # c: ChunkKV chunk length; 1 = token-level (old behaviour)
    loop_decay: float = 1.0        # 1.0 = flat; <1 shrinks budget with loop depth
    sink_tokens: int = 4           # attention sink tokens always kept
    min_prompt_tokens: int = 256   # compress only when S - ws >= this (guard)
    enabled: bool = True
    # ---- loop-anchored allocator (our own, replaces DynamicKV's Phi) ----
    allocator: str = "block"       # block | anchor | overlap  (see module docs)
    top_frac: float = 0.10         # size (as a fraction of S) of the "important
                                  # set" whose cross-loop overlap is measured
    budget_scale: float = 1.0      # alpha: global multiplier on the overlap
                                  # budget (1.0 = pure similarity-driven)
    theta: float = 0.9             # coverage target for the anchor allocator
    budget_mode: str = "need"      # "need" (rho is an outcome) | "rho" (rescaled)
    # ---- "js" allocator: Jensen-Shannon divergence between loops -----------
    # ONE rule for every (loop, layer) unit, no threshold and no per-loop
    # special case:
    #
    #     d_u   = JS( s_(t-1,l), s_(t,l) )     how far THIS loop moved
    #     rho_u = d_u / D1                     D1 = running median of the first
    #                                          transition's JS (per sample)
    #     r_u   = clip( kappa * rho_u^gamma, js_floor, 1 )
    #     budget_u = r_u * (S - w_s)
    #
    # Reading of the three knobs:
    # * first compressible loop has rho ~ 1 -> r ~ ``js_kappa``.  kappa IS that
    #   loop's retention: raise it to give Loop-2 more KV, no other code path;
    # * later loops have rho << 1 once attention froze -> they land on
    #   ``js_floor``, and stay proportionally higher wherever attention is
    #   still re-routing (retrieval-style tasks), again with no explicit test;
    # * ``js_gamma`` shapes the decay from kappa down to the floor: gamma = 1
    #   falls to the floor ("shrink"), gamma < 1 keeps later loops wide.
    # Everything else (window, sink, chunk size, b_min) is selection-only.
    js_gamma: float = 1.0          # exponent on rho: 1 = fall to floor fast,
                                   # < 1 = later loops decay more gently
    js_kappa: float = 1.0          # retention of the first compressed loop
                                   # (rho ~ 1 there), i.e. its KV budget
    js_floor: float = 0.10         # retention floor for frozen loops/layers
    # ---- "fixed" allocator: a hand-written per-loop rule -------------------
    # Comma separated retention ratio per loop, e.g. "1,1,0.2,0.1".
    # Empty -> fall back to [1.0]*keep_loops + [compressed_ratio()]*(H-K).
    loop_ratios: str = ""
    # "prev"  : anchor of loop t is loop t-1 (adjacent-loop convergence)
    # "loop1" : anchor of every loop is loop 1 (fixed reference)
    anchor_source: str = "prev"

    def describe(self) -> str:
        head = (f"DynaLoop({self.allocator}, K={self.keep_loops}, "
                f"ws={self.window_size}, b_min={self.min_budget_ratio}, "
                f"m={self.block_size}, c={self.chunk_size}, "
                f"min_prompt={self.min_prompt_tokens}")
        if self.allocator == "js":
            return (head + f", kappa={self.js_kappa}, gamma={self.js_gamma}, "
                    f"floor={self.js_floor})")
        if self.allocator == "fixed":
            return head + f", loop_ratios={self.loop_ratios!r})"
        if self.allocator == "block":
            return head + f", rho={self.global_ratio}, decay={self.loop_decay})"
        return head + ")"

    def compressed_ratio(self) -> float:
        """Retention ratio each compressed loop must realise to hit ``rho``."""
        h, k = self.num_loops, self.keep_loops
        if h <= k:
            return 1.0
        r = (self.global_ratio * h - k) / (h - k)
        return float(max(r, self.min_budget_ratio))

    def loop_ratio(self, ut: int) -> float:
        """Per-loop retention ratio (1.0 for the first ``keep_loops`` loops)."""
        if ut < self.keep_loops or not self.enabled:
            return 1.0
        base = self.compressed_ratio()
        depth = ut - self.keep_loops
        return float(max(base * (self.loop_decay ** depth), self.min_budget_ratio))


def window_query_scores(
    q_window: torch.Tensor,      # [B, H, w, Dk] queries of the last w positions (RoPE applied)
    k_full: torch.Tensor,        # [B, H, S, Dk] keys of the whole prompt
    scaling: float,
    causal: bool = True,
    q_positions: Optional[torch.Tensor] = None,   # [B, w] absolute positions
) -> torch.Tensor:
    """Per-key-position importance, DynamicKV's ``Pooling(A[:, ws])``.

    ``sum over window queries`` then ``sum over heads`` of the softmax
    attention probabilities -- identical to what DynamicKV pools, computed
    from the real (causal) attention of the window on the prompt.

    Returns ``[B, S]``; reduction is over the head and window axes only, so
    batching does not mix samples.  ``q_positions`` lets a right-padded batch
    place each sample's window at its own true end of sequence.
    """
    # [B, H, w, S]
    logits = torch.matmul(q_window, k_full.transpose(-1, -2)) * scaling
    if causal:
        s = k_full.shape[-2]
        w = q_window.shape[-2]
        b = q_window.shape[0]
        if q_positions is None:
            q_positions = (torch.arange(s - w, s, device=logits.device)
                           .unsqueeze(0).expand(b, -1))
        if q_positions.dim() == 1:      # single sample -> [1, w]
            q_positions = q_positions.unsqueeze(0)
        # [1, 1, 1, S] vs [B, 1, w, 1] -> broadcast to [B, 1, w, S]
        pos_k = torch.arange(s, device=logits.device)[None, None, None, :]
        logits = logits.masked_fill(pos_k > q_positions[:, None, :, None],
                                    torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.sum(dim=(1, 2))          # [B, S]


def _forced_mask(seq_len: int, window_size: int, sink_tokens: int,
                 device) -> torch.Tensor:
    """bool[S]: True where a position is *eligible* for budgeted selection.

    Sink tokens and the recent window are always kept, so they are excluded
    from the budgeted candidate set (they must not double-count, and they must
    not bias which chunk looks "important").
    """
    valid = torch.ones(seq_len, dtype=torch.bool, device=device)
    forced = set(range(min(sink_tokens, seq_len)))
    win_start = max(0, seq_len - window_size)
    forced.update(range(win_start, seq_len))
    if forced:
        valid[torch.tensor(sorted(forced), device=device)] = False
    return valid


def chunk_scores(
    scores: torch.Tensor,          # [S] importance per key position
    valid: torch.Tensor,           # [S] bool, positions eligible for selection
    chunk_size: int,
) -> torch.Tensor:
    """ChunkKV (arXiv 2502.00299, Alg. 1): per-chunk score = sum over its tokens.

    Chunks are aligned on the **full** key sequence (``0, c, 2c, ...``), exactly
    as in the paper, so chunk boundaries do not depend on which positions were
    masked out. Positions outside ``valid`` contribute 0.
    """
    seq_len = scores.shape[0]
    c = max(int(chunk_size), 1)
    n_chunks = (seq_len + c - 1) // c
    cid = torch.arange(seq_len, device=scores.device) // c
    out = torch.zeros(n_chunks, device=scores.device, dtype=torch.float32)
    return out.index_add(0, cid, scores.float() * valid.float())


def keep_indices_for_unit(
    scores: torch.Tensor,          # [S] importance per key position
    budget: int,                   # number of prompt tokens to keep
    seq_len: int,
    window_size: int,
    sink_tokens: int,
    chunk_size: int = 1,           # c: contiguous block length; 1 = token-level
) -> torch.Tensor:
    """Indices kept for one unit: sink + recent window + top-budget.

    With ``chunk_size = c > 1`` the top-budget part is replaced by ChunkKV's
    block selection: the sequence is cut into contiguous chunks of ``c`` tokens,
    the ``k = floor(budget / c)`` highest-scoring chunks are kept **whole** and
    returned in their original order. Chunks are never split, so what survives
    is always a set of contiguous runs of length ``c`` (plus the always-kept
    sink and window).
    """
    device = scores.device
    forced = set(range(min(sink_tokens, seq_len)))
    win_start = max(0, seq_len - window_size)
    forced.update(range(win_start, seq_len))

    budget = int(max(budget, 0))
    valid = _forced_mask(seq_len, window_size, sink_tokens, device)
    cand_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if cand_idx.numel() == 0:
        return torch.arange(seq_len, device=device)

    c = max(int(chunk_size), 1)
    k = budget // c

    if c <= 1 or k <= 0:
        # token-level fallback (also used when the budget is below one chunk)
        n_sel = min(budget, int(cand_idx.numel()))
        if n_sel > 0:
            top = torch.topk(scores[cand_idx], k=n_sel, sorted=False).indices
            selected = cand_idx[top]
        else:
            selected = cand_idx[:0]
    else:
        cs = chunk_scores(scores, valid, c)
        k = min(k, cs.shape[0])
        top_chunks = torch.topk(cs, k=k, sorted=False).indices
        # keep whole chunks, in original sequence order (ChunkKV Alg. 1).
        # Vectorised: mark the winning chunk ids, then gather per token --
        # identical to `mask |= (cid == ch)` for every selected chunk, but
        # without the O(k) Python loop (k ~ 140), which dominated prefill.
        sel = torch.zeros(cs.shape[0], dtype=torch.bool, device=device)
        sel[top_chunks] = True
        cid = torch.arange(seq_len, device=device) // c
        mask = sel[cid] & valid
        selected = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    keep = torch.cat([torch.tensor(sorted(forced), device=device), selected]).unique()
    return keep


def anchor_coverage_need(
    ref: torch.Tensor,        # [S] loop-1 importance of the SAME layer
    cur: torch.Tensor,        # [S] importance of this (loop, layer)
    theta: float,
) -> int:
    """Loop-anchored budget: how many tokens are needed for THIS loop to be
    ``theta``-covered **when the tokens are offered in the anchor loop's order**
    (``anchor_source``: ``"prev"`` -> loop t-1, ``"loop1"`` -> loop 1).

    ``need = min { k : sum_{i<=k} cur[pi_ref(i)] >= theta }``

    ``theta`` is a **mass-coverage target, not a token-overlap target**:

    * numerator/denominator are both ``cur`` (this loop's *own* normalised
      attention mass) -- we accumulate ``cur`` along ``ref``'s ranking;
    * so ``theta = 0.9`` reads as "keep enough tokens that 90% of this loop's
      own attention mass is still reachable", measured by walking down the
      anchor's importance order;
    * it says nothing directly about *how many of the same tokens* the two
      loops attend to.  That quantity is ``overlap_ratio`` (the ``overlap``
      allocator) and is a genuinely different number -- e.g. on
      ``math500_d0`` layer 23, loop4 vs loop3 at ``theta = 0.9``:
      need = 912 tokens, same-token overlap = 0.940, self-coverage = 0.901.

    A loop that has converged onto the anchor's focus is covered by the very
    head of the anchor's ranking and needs almost nothing; a loop that is
    still re-routing its attention finds its mass far down the anchor's
    ranking and therefore needs a lot.  The budget is a *measurement* of the
    cross-loop convergence, not a heuristic -- and because the actual
    selection in :func:`keep_indices_for_unit` is done with ``cur``'s own
    ranking, keeping ``need`` tokens always retains **at least** ``theta`` of
    ``cur``'s mass (self-ranked top-k dominates any other k-subset).

    Caveat: the theta -> need mapping is extremely sensitive to how *flat*
    the score vector is.  On a sharp single-query distribution ``theta=0.9``
    costs 5.7% of S; on the flat window-pooled score the engine actually uses
    the same ``theta=0.9`` costs 38.8% of S (same sample, same layer).
    """
    order = torch.argsort(ref, descending=True)
    cum = torch.cumsum(cur[order].float(), dim=0)
    total = cum[-1].clamp_min(1e-12)
    frac = cum / total
    k = int(torch.searchsorted(frac, float(theta)).clamp(max=frac.numel() - 1)) + 1
    return int(min(max(k, 1), cur.numel()))


def allocate_anchor_budgets(
    block_units: Sequence[Tuple[int, int]],
    scores: Dict[Tuple[int, int], torch.Tensor],
    seq_len: int,
    cfg: DynaLoopConfig,
    refs: Dict[int, torch.Tensor],     # layer -> loop-1 score vector [S]
) -> Dict[Tuple[int, int], int]:
    """Per-unit budget from a *reference loop's* ranking (no cross-layer coupling).

    The reference is loop 1 when ``cfg.anchor_source == "loop1"`` and the
    **immediately preceding loop** when it is ``"prev"`` -- the caller decides
    by simply storing whichever score vector it wants into ``refs``.

    Every reference/anchored loop below ``keep_loops`` stays full; the others
    get ``need_u(theta)`` tokens, floored by ``b_min`` and capped by
    ``seq_len``.  Unlike :func:`allocate_block_budgets` this needs no block:
    each unit is decided on its own, so ``block_size`` may be 1 and each unit
    can be compressed the moment it is produced.

    With ``anchor_source = "prev"`` the measured quantity is *one loop step of
    attention re-routing*: need is small exactly where the attention map has
    already frozen between loop t-1 and loop t, and large where it is still
    moving.  Because ``keep_indices_for_unit`` then selects with ``cur``'s own
    ranking, keeping ``need`` tokens always retains at least ``theta`` of the
    current loop's own mass.
    """
    prompt_tokens = max(seq_len - cfg.window_size, 1)
    floor = max(int(prompt_tokens * cfg.min_budget_ratio), 1)
    out: Dict[Tuple[int, int], int] = {}
    keep_full = (not cfg.enabled) or (prompt_tokens < int(cfg.min_prompt_tokens))
    for u in block_units:
        ut, layer = u
        if keep_full or ut < max(int(cfg.keep_loops), 1):
            out[u] = seq_len
            continue
        ref = refs.get(layer)
        if ref is None or ref.numel() != seq_len:
            out[u] = seq_len            # no anchor yet -> stay safe
            continue
        out[u] = int(min(max(anchor_coverage_need(ref, scores[u], cfg.theta),
                             floor), seq_len))
    return out


def overlap_ratio(ref: torch.Tensor, cur: torch.Tensor, k: int) -> float:
    """Jaccard-style overlap of the two loops' important-token sets.

    ``|TopK(ref, k) ∩ TopK(cur, k)| / k`` -- 1.0 means this loop looks at
    exactly the same tokens as its anchor (converged), 0 means it has
    completely re-routed.  Measured on the *sets*, so it is insensitive to
    how flat the underlying score distribution is (unlike the coverage-need
    formulation, which degenerates on a flat pooled score).
    """
    k = int(max(min(k, ref.numel()), 1))
    a = torch.topk(ref.float(), k=k, sorted=False).indices
    b = torch.topk(cur.float(), k=k, sorted=False).indices
    return float((a.unsqueeze(1) == b.unsqueeze(0)).any(dim=1).sum()) / k


def allocate_overlap_budgets(
    block_units: Sequence[Tuple[int, int]],
    scores: Dict[Tuple[int, int], torch.Tensor],
    seq_len: int,
    cfg: DynaLoopConfig,
    refs: Dict[int, torch.Tensor],     # layer -> previous-loop score vector [S]
) -> Dict[Tuple[int, int], int]:
    """Budget = share of tokens whose importance is **new** vs the previous loop.

    ``r_u = 1 - overlap_k(s_(t-1,l), s_(t,l))``  ->  ``budget_u = r_u * (S - w_s)``

    Reading: after the first loops have picked out the important tokens, a
    later loop only needs to *add* the tokens it newly discovered; everything
    already agreed upon is cheap.  Freeze (overlap -> 1) therefore drives the
    budget to the floor, while a genuine re-routing (overlap low) opens it up.
    """
    prompt_tokens = max(seq_len - cfg.window_size, 1)
    floor = max(int(prompt_tokens * cfg.min_budget_ratio), 1)
    k = max(int(round(cfg.top_frac * seq_len)), 1)
    out: Dict[Tuple[int, int], int] = {}
    keep_full = (not cfg.enabled) or (prompt_tokens < int(cfg.min_prompt_tokens))
    for u in block_units:
        ut, layer = u
        if keep_full or ut < max(int(cfg.keep_loops), 1):
            out[u] = seq_len
            continue
        ref = refs.get(layer)
        if ref is None or ref.numel() != seq_len:
            out[u] = seq_len            # no anchor yet -> stay safe
            continue
        r = 1.0 - overlap_ratio(ref, scores[u], k)
        out[u] = int(min(max(int(round(cfg.budget_scale * r * prompt_tokens)),
                             floor), seq_len))
    return out


def js_divergence(ref: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (nats) between two importance distributions.

    Both vectors are normalised first, so the result is scale free -- it does
    not matter that the engine's score is pooled over heads and window queries.
    ``JS = 0`` means the two loops rank the context identically, ``JS = ln 2``
    means they are completely disjoint.  Because it compares the *whole*
    distribution (not just a top-k set), it picks up re-routing that happens
    in the tail, which is exactly what ``overlap`` misses.
    """
    p = ref.float().clamp_min(1e-12)
    q = cur.float().clamp_min(1e-12)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a: (a * (a.log() - m.log())).sum()   # noqa: E731
    return 0.5 * kl(p) + 0.5 * kl(q)


def allocate_js_budgets(
    block_units: Sequence[Tuple[int, int]],
    scores: Dict[Tuple[int, int], torch.Tensor],
    seq_len: int,
    cfg: DynaLoopConfig,
    refs: Dict[int, torch.Tensor],     # layer -> previous-loop score vector [S]
    js_stats: Optional[Dict] = None,   # per-sample {"d1": [JS of transition 1]}
) -> Dict[Tuple[int, int], int]:
    """Per-(loop, layer) budget = how far this unit's attention moved.

    For every unit ``u = (t, l)`` whose anchor is layer ``l``'s score vector
    of the previous loop (``anchor_source == "prev"``):

        d_u   = JS( s_(t-1,l), s_(t,l) )        how far this loop moved (nats)
        rho_u = d_u / D1                        D1 = running median of the
                                                first transition's JS
        r_u   = clip( kappa * rho_u^gamma , js_floor , 1 )
        budget_u = r_u * (S - w_s)

    One formula, no threshold, no per-loop branch:

    * the first compressible loop has ``rho ~ 1`` -> ``r ~ js_kappa``: kappa
      directly sets that loop's retention (and only that behaves like a knob
      for it -- everything else is a consequence of the measured JS);
    * later loops have ``rho << 1`` once attention froze -> they land on
      ``js_floor``; where attention keeps re-routing, rho stays high and they
      keep proportionally more, for free;
    * ``js_gamma`` interpolates between "fall to the floor" (gamma = 1) and
      "stay wide after the first loop" (gamma < 1).

    Because ``d_u`` is kept per layer (not pooled over the loop) the budget is
    genuinely (loop, layer) granular: layers that are still moving keep more
    than layers that already froze, inside the same loop.

    Measured on Ouro-1.4B (9 samples, S ~ 2000, pooled window score):
    loop1->2 JS = 0.377, loop2->3 = 0.0087, loop3->4 = 0.0015, i.e. rho drops
    by ~25x per transition on convergent tasks, which is exactly the dynamic
    range this mapping consumes.
    """
    prompt_tokens = max(seq_len - cfg.window_size, 1)
    floor = max(int(prompt_tokens * cfg.min_budget_ratio), 1)
    first_t = max(int(cfg.keep_loops), 1)
    if js_stats is None:
        js_stats = {}
    out: Dict[Tuple[int, int], int] = {}

    if (not cfg.enabled) or prompt_tokens < int(cfg.min_prompt_tokens):
        return {u: seq_len for u in block_units}          # guard: keep full

    # 1. per-(loop, layer) displacement against the previous loop
    dists: Dict[Tuple[int, int], float] = {}
    for u in block_units:
        ut, layer = u
        if ut < first_t:
            continue                                      # full by policy
        ref = refs.get(layer)
        if ref is None or ref.numel() != seq_len:
            continue                                      # no anchor -> full
        dists[u] = float(js_divergence(ref, scores[u]))

    # 2. reference scale D1 = running median of the first transition
    if block_units[0][0] == first_t and dists:
        js_stats.setdefault("d1", []).extend(dists.values())
    d1 = js_stats.get("d1")
    d1_ref = sorted(d1)[len(d1) // 2] if d1 else None

    # 3. budget
    gamma = max(float(cfg.js_gamma), 1e-6)
    kappa = float(cfg.js_kappa)
    lower = float(cfg.js_floor)
    for u in block_units:
        if u not in dists or d1_ref is None:
            out[u] = seq_len                              # not measurable yet
            continue
        rho = max(dists[u] / max(d1_ref, 1e-12), 0.0)
        r = float(min(max(kappa * (rho ** gamma), lower), 1.0))
        out[u] = seq_len if r >= 1.0 else int(
            min(max(int(round(r * prompt_tokens)), floor), seq_len))
    return out


def parse_loop_ratios(cfg: DynaLoopConfig) -> List[float]:
    """Per-loop retention ratio for the ``fixed`` allocator.

    Honours ``cfg.loop_ratios`` when set (e.g. ``"1,1,0.2,0.1"``); otherwise
    falls back to the plain DynaLoop rule -- first ``keep_loops`` loops full,
    every later loop at ``compressed_ratio()``.
    """
    h = max(int(cfg.num_loops), 1)
    txt = (cfg.loop_ratios or "").strip()
    if txt:
        vals = [float(x) for x in txt.split(",") if x.strip() != ""]
        if len(vals) >= h:
            return [min(max(v, 0.0), 1.0) for v in vals[:h]]
        vals = vals + [vals[-1]] * (h - len(vals))
        return [min(max(v, 0.0), 1.0) for v in vals]
    k = max(int(cfg.keep_loops), 0)
    base = cfg.compressed_ratio()
    return [1.0] * min(k, h) + [base] * max(h - k, 0)


def allocate_fixed_budgets(
    block_units: Sequence[Tuple[int, int]],
    scores: Dict[Tuple[int, int], torch.Tensor],
    seq_len: int,
    cfg: DynaLoopConfig,
    refs: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[Tuple[int, int], int]:
    """Rule-based budget: a fixed retention ratio per loop, nothing adaptive.

    ``budget_u = ratio[ut] * (S - w_s)``  (the always-kept sink + window are
    added later by :func:`keep_indices_for_unit`, as for every allocator).

    This is the control condition for the loop-driven allocators: same
    per-loop numbers, but written down by hand instead of measured from the
    attention.  Comparing it against ``overlap`` at equal realised retention
    isolates what the adaptivity actually buys.
    """
    prompt_tokens = max(seq_len - cfg.window_size, 1)
    ratios = parse_loop_ratios(cfg)
    keep_full = (not cfg.enabled) or (prompt_tokens < int(cfg.min_prompt_tokens))
    out: Dict[Tuple[int, int], int] = {}
    for u in block_units:
        ut = u[0]
        if keep_full:
            out[u] = seq_len
            continue
        r = ratios[ut] if ut < len(ratios) else ratios[-1]
        if r >= 1.0:
            out[u] = seq_len
        else:
            out[u] = int(min(max(int(round(r * prompt_tokens)), 1), seq_len))
    return out


def allocate_block_budgets(
    block_units: Sequence[Tuple[int, int]],   # [(ut, layer), ...] in the block
    scores: Dict[Tuple[int, int], torch.Tensor],
    seq_len: int,
    cfg: DynaLoopConfig,
    n_processed: int,
) -> Dict[Tuple[int, int], int]:
    """DynamicKV progressive update over one block of consecutive units.

    ``C_u = Phi(TopK(A, (w_t - w_s) * n))`` -- the count of globally
    top-ranked entries falling into each unit -- then
    ``Z_u = B_u * C_u / max(C)`` renormalised to the block budget and
    floored by ``b_min``.

    When ``cfg.chunk_size = c > 1`` the ranking entry is the **chunk**
    (ChunkKV Alg. 1) instead of the token: each unit contributes
    ``ceil(S / c)`` entries whose value is the sum of that chunk's token
    scores, and ``Phi(TopK(...))`` counts how many globally top-ranked chunks
    land in each unit. The returned budget is still a token count; the caller
    converts it to ``k = budget // c`` whole chunks.
    """
    prompt_tokens = max(seq_len - cfg.window_size, 1)  # w_t - w_s
    # Per-unit maximum budget B_u = (w_t - w_s) * r(loop).
    b_u = {u: max(int(prompt_tokens * cfg.loop_ratio(u[0])), 1) for u in block_units}

    if not cfg.enabled:
        return {u: seq_len for u in block_units}

    # ---- length-aware activation guard (our addition) -------------------
    # Token selection only pays off when the *selectable* region is large
    # enough to contain redundancy.  On a dense short prompt (e.g. a 106-token
    # GSM8K question) sink + window already cover ~64% of the sequence, so a
    # 10% budget drops ~36% of the actual problem statement and accuracy
    # collapses (measured: GSM8K 0.46 -> 0.26).  Below the threshold we keep
    # every token: no saving, but also no loss.
    if prompt_tokens < int(cfg.min_prompt_tokens):
        return {u: seq_len for u in block_units}

    c = max(int(cfg.chunk_size), 1)
    if c > 1:
        valid = _forced_mask(seq_len, cfg.window_size, cfg.sink_tokens,
                             scores[block_units[0]].device)
        stacked = torch.stack([chunk_scores(scores[u], valid, c)
                               for u in block_units], dim=0)     # [n, C]
        topk_total = max(int(prompt_tokens) // c, 1) * len(block_units)
    else:
        stacked = torch.stack([scores[u] for u in block_units], dim=0)  # [n, S]
        topk_total = int(prompt_tokens) * len(block_units)      # (w_t - w_s) * n

    topk_total = min(topk_total, stacked.numel())
    flat = stacked.reshape(-1)
    top_vals, top_pos = torch.topk(flat, k=topk_total, sorted=False)
    unit_of = (top_pos // stacked.shape[1]).long()
    counts = torch.bincount(unit_of, minlength=len(block_units)).float()  # C_u
    if counts.max() <= 0:
        counts = torch.ones_like(counts)

    z = torch.tensor([b_u[u] for u in block_units], dtype=torch.float32,
                     device=stacked.device) * counts / counts.max()

    # renormalise to the block budget B = (w_t - w_s) * L * mean loop ratio
    target = sum(b_u.values())
    if z.sum() > 0:
        z = z / z.sum() * target

    floor = int(prompt_tokens * cfg.min_budget_ratio)
    out = {}
    for i, u in enumerate(block_units):
        out[u] = int(max(min(int(round(float(z[i]))), seq_len), min(floor, seq_len)))
    return out
