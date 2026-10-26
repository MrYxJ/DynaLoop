"""CPU-only unit tests for the budget allocator (no GPU, no model needed).

Runs either way::

    python tests/test_policy.py          # no test framework required
    python -m pytest tests -q            # if pytest is installed
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynaloop.policy import (DynaLoopConfig, allocate_js_budgets,  # noqa: E402
                             chunk_scores, js_divergence,
                             keep_indices_for_unit, window_query_scores)

torch.manual_seed(0)

SEQ = 512
UNITS = [(t, 0) for t in range(4)]          # one layer, four loops


def _scores(sharp: float):
    """Make one importance vector whose peakiness is controlled."""
    x = torch.arange(SEQ, dtype=torch.float32)
    return torch.exp(-sharp * x / SEQ)


def test_js_zero_for_identical():
    s = _scores(1.0)
    assert float(js_divergence(s, s)) < 1e-6


def test_js_increases_with_displacement():
    s0 = _scores(1.0)
    small = torch.roll(s0, 8)
    big = torch.roll(s0, 128)
    assert js_divergence(s0, small) < js_divergence(s0, big)


def test_js_is_bounded_by_ln2():
    # disjoint one-hot distributions attain ln 2
    p = torch.zeros(8)
    p[0] = 1.0
    q = torch.zeros(8)
    q[1] = 1.0
    assert abs(float(js_divergence(p, q)) - 0.6931) < 0.01


def test_keep_indices_always_includes_sink_and_window():
    s = _scores(1.0)
    keep = keep_indices_for_unit(s, budget=10, seq_len=SEQ, window_size=64,
                                 sink_tokens=4, chunk_size=1)
    assert set(range(4)).issubset(set(keep.tolist()))            # sink
    assert set(range(SEQ - 64, SEQ)).issubset(set(keep.tolist()))  # window
    assert keep.numel() <= 10 + 4 + 64


def test_chunk_selection_keeps_whole_chunks():
    """A selected chunk survives as a whole run of ``c`` consecutive tokens."""
    c, ws, sink = 5, 64, 4
    s = _scores(0.5)
    keep = keep_indices_for_unit(s, budget=100, seq_len=SEQ, window_size=ws,
                                 sink_tokens=sink, chunk_size=c)
    kept = set(keep.tolist())
    forced = set(range(sink)) | set(range(SEQ - ws, SEQ))
    for start in range(0, SEQ, c):
        block = [i for i in range(start, min(start + c, SEQ))]
        if any(i in forced for i in block):
            continue                      # partially forced chunks are skipped
        inside = sum(1 for i in block if i in kept)
        assert inside in (0, len(block)), (start, inside, len(block))


def test_chunk_scores_sum_within_chunk():
    s = torch.arange(10, dtype=torch.float32)
    valid = torch.ones(10, dtype=torch.bool)
    cs = chunk_scores(s, valid, 5)
    assert cs.tolist() == [10.0, 35.0]


def test_window_query_scores_shape_and_head_reduction():
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, SEQ, 16)
    s = window_query_scores(q, k, scaling=0.25, causal=True)
    assert s.shape == (1, SEQ)


def test_short_prompt_guard_keeps_everything():
    seq = 100
    cfg = DynaLoopConfig(keep_loops=1, min_prompt_tokens=192,
                         allocator="js", js_kappa=0.5, js_floor=0.10)
    units = [(t, 0) for t in range(4)]
    sc = {u: _scores(1.0) for u in units}
    out = allocate_js_budgets(units, sc, seq, cfg, refs={0: _scores(1.0)})
    assert all(v == seq for v in out.values()), out


def test_first_compressed_loop_gets_about_kappa():
    """``rho = 1`` at the first compressed loop, so its ratio lands on kappa.

    Loops are 0-indexed here: ``keep_loops=1`` keeps unit ``(0, l)`` full, so
    ``(1, l)`` is the first loop whose budget is decided by the rule.
    """
    cfg = DynaLoopConfig(keep_loops=1, window_size=64, min_prompt_tokens=192,
                         allocator="js", js_kappa=0.5, js_gamma=1.0,
                         js_floor=0.10)
    seq = 1000
    prompt = seq - cfg.window_size
    base = torch.rand(seq) + 0.1
    moved = torch.roll(base, 64)                 # a genuinely re-routed loop
    units = [(t, 0) for t in range(4)]
    sc = {(1, 0): moved, (2, 0): base.clone(), (3, 0): base.clone()}

    # first pass: the allocator self-calibrates D1 from loop 0 -> loop 1
    stats: dict = {}
    allocate_js_budgets([(1, 0)], sc, seq, cfg, refs={0: base},
                        js_stats=stats)
    assert stats.get("d1"), "D1 calibration must be recorded"

    out = allocate_js_budgets(units, sc, seq, cfg, refs={0: base},
                              js_stats=stats)
    assert out[(0, 0)] == seq                    # full by policy
    assert abs(out[(1, 0)] / prompt - 0.5) < 0.05, out
    # the displacement from loop 1 is rho ~ 1 -> r ~ kappa; loops that did not
    # move at all collapse onto the floor
    assert out[(2, 0)] <= int(0.11 * prompt) + 1, out
    assert out[(3, 0)] <= int(0.11 * prompt) + 1, out


def test_gamma_shapes_the_decay_from_kappa_to_floor():
    """Same half-moved unit: gamma < 1 keeps it wider than gamma = 1."""
    seq, prompt = 1000, 1000 - 64
    base = torch.rand(seq) + 0.1
    d1 = js_divergence(base, torch.roll(base, 64)).item()

    def _mix(w: float) -> torch.Tensor:
        return (1 - w) * base + w * torch.roll(base, 16)

    # bisect the mixing weight until JS(base, mix(w)) == d1 / 2, i.e. rho = 0.5
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if js_divergence(base, _mix(mid)).item() < d1 / 2:
            lo = mid
        else:
            hi = mid
    mid = _mix(hi)
    assert abs(js_divergence(base, mid).item() / d1 - 0.5) < 0.01

    got = {}
    for gamma in (0.5, 1.0):
        cfg = DynaLoopConfig(keep_loops=1, window_size=64,
                             min_prompt_tokens=192, allocator="js",
                             js_kappa=0.5, js_gamma=gamma, js_floor=0.10)
        out = allocate_js_budgets([(1, 0)], {(1, 0): mid}, seq, cfg,
                                  refs={0: base}, js_stats={"d1": [d1]})
        got[gamma] = out[(1, 0)] / prompt
    assert got[0.5] > got[1.0] > 0.10, got      # shallower decay keeps more


def _main() -> int:
    """Tiny runner so the suite works without pytest installed."""
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:                                    # noqa: BLE001
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
