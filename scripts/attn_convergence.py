"""Quantify how fast attention stabilises across the loops of a looped LM.

This is the *motivation* measurement behind DynaLoop: it turns the informal
claim "later loops look at the same tokens" into numbers, and it is the reason
the allocator is driven by inter-loop displacement instead of a fixed schedule.

For every captured sample (produced by :mod:`scripts.attn_loop_probe`) the
script loads ``p_last[loop, layer, token]`` -- the attention distribution of the
**last** prompt token's query, i.e. the distribution that decides the first
generated token -- and computes, per loop:

``js_vs_final``
    Jensen-Shannon divergence to the *same layer* in the final loop
    (averaged over the last two layers).
``top10_overlap``
    Fraction of the top-10% most-attended keys that coincide with the final
    loop's set.  Unlike JS, this is invariant to how the residual mass is
    spread, so it measures *which* tokens, not how much weight.
``js_adjacent_loop``
    Drift between consecutive loops at a fixed layer, averaged over all 24
    layers -- the quantity DynaLoop's budget rule is built on.
``js_adjacent_layer``
    Drift between consecutive layers inside the final loop.  This is the
    natural yardstick: if cross-loop drift is far below cross-layer drift, then
    a (loop, layer) pair is mostly redundant with its own previous loop, while
    consecutive layers are genuinely different functions.
``js_layer0_last``
    Distance between the first and the last layer of the final loop: the scale
    of a distribution change that is *not* a redundancy.
``mass80_tokens``
    Number of keys carrying 80% of the attention mass in the final loop.

Run::

    python scripts/attn_convergence.py \
        --attn-dir results/attn --attn-dir results/attn_niah1024

which writes ``results/attn_convergence.json`` (consumed by
``scripts/make_figures.py``).  The script is CPU-only and needs no model.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-12

TASK_NAMES = {"math500": "MATH-500", "gsm8k": "GSM8K", "niah": "NIAH-Multi"}


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #
def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p / p.sum(), EPS, 1.0)


def js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats."""
    p, q = _norm(p), _norm(q)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum()
                 + 0.5 * (q * np.log(q / m)).sum())


def topk_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    a = set(np.argsort(p)[-k:])
    b = set(np.argsort(q)[-k:])
    return len(a & b) / float(k)


def mass_tokens(p: np.ndarray, frac: float = 0.8) -> int:
    s = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    return int(np.searchsorted(np.cumsum(s), frac) + 1)


def loop_metrics(P: np.ndarray, top_layers=(22, 23), frac: float = 0.10):
    """``P``: ``[U, L, T]`` attention distributions.  Returns per-loop scalars."""
    U, Ln, T = P.shape
    ref = P[U - 1]
    k = max(1, int(round(T * frac)))
    out = {
        "js_vs_final": [float(np.mean([js(P[u, i], ref[i]) for i in top_layers]))
                        for u in range(U)],
        "top10_overlap": [float(np.mean([topk_overlap(P[u, i], ref[i], k)
                                         for i in top_layers]))
                          for u in range(U)],
        "js_adjacent_loop": [float(np.mean([js(P[u, i], P[u + 1, i])
                                            for i in range(Ln)]))
                             for u in range(U - 1)],
        "js_adjacent_layer_mean": float(np.mean([js(ref[i], ref[i + 1])
                                                 for i in range(Ln - 1)])),
        "js_layer0_last": js(ref[0], ref[Ln - 1]),
        "mass80_tokens": float(np.mean([mass_tokens(ref[i], 0.8)
                                        for i in top_layers])),
        "n_tokens": int(T),
    }
    return out


# --------------------------------------------------------------------------- #
def _agg(samples: list, key: str) -> dict | None:
    """Mean / std over samples for a scalar or per-loop list."""
    vals = [s[key] for s in samples if key in s]
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist()}


def _collect(dirs: list[str]) -> dict:
    buckets: dict[str, list] = {}
    for d in dirs:
        for npz in sorted(glob.glob(os.path.join(d, "*_attn.npz"))):
            meta_p = npz.replace("_attn.npz", "_meta.json")
            if not os.path.exists(meta_p):
                continue
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            task = TASK_NAMES.get(str(meta.get("task", "")).lower(),
                                  str(meta.get("task", "unknown")))
            with np.load(npz) as z:
                if "p_last" not in z:
                    continue
                P = z["p_last"].astype(np.float32)
            m = loop_metrics(P)
            m["sample"] = os.path.splitext(os.path.basename(npz))[0]
            buckets.setdefault(task, []).append(m)
            print("  %-22s %s" % (task, m["sample"]), flush=True)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn-dir", action="append", default=None,
                    help="directory holding <tag>_attn.npz + <tag>_meta.json "
                         "(repeatable)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "attn_convergence.json"))
    args = ap.parse_args()

    dirs = args.attn_dir or [os.path.join(ROOT, "results", "attn")]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("no --attn-dir found; nothing to do", file=sys.stderr)
        return 1

    buckets = _collect(dirs)
    if not buckets:
        print("no attention dumps found", file=sys.stderr)
        return 1

    tasks = {}
    for task, samples in sorted(buckets.items()):
        tasks[task] = {
            "n_samples": len(samples),
            "js_vs_final": _agg(samples, "js_vs_final"),
            "top10_overlap": _agg(samples, "top10_overlap"),
            "js_adjacent_loop": _agg(samples, "js_adjacent_loop"),
            "js_adjacent_layer_mean": _agg(samples, "js_adjacent_layer_mean"),
            "js_layer0_last": _agg(samples, "js_layer0_last"),
            "mass80_tokens": _agg(samples, "mass80_tokens"),
            "n_tokens": _agg(samples, "n_tokens"),
        }

    payload = {
        "note": ("JS divergence (nats) between attention distributions of the "
                 "last prompt token; top-10% overlap measured against the "
                 "final loop"),
        "tasks": tasks,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print("saved ->", args.out)
    for task, v in tasks.items():
        print("  %-12s n=%d  js_vs_final=%s" %
              (task, v["n_samples"],
               np.round(v["js_vs_final"]["mean"], 4).tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
