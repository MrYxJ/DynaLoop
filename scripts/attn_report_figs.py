"""Regenerate the attention-analysis figures from ``*_attn.npz`` dumps.

These are the *diagnostic* figures behind the motivation: they visualise, on a
single captured sample, what the aggregate statistics in
``results/attn_convergence.json`` summarise across samples.

    python scripts/attn_report_figs.py \
        --attn-dir results/attn --main math500_d0 \
        --niah-dir results/attn_niah1024

Sources (per sample, produced by ``scripts/attn_loop_probe.py``):
``p_last[loop, layer, token]`` -- attention of the last prompt token;
``ph_last[loop, layer, head, token]`` -- the same, per head;
``maps[loop, head, q, k]`` -- down-sampled head x query x key map of one layer.

Outputs go to ``docs/figures/attn_*.png``; the per-sample table used in the
README is written to ``results/attn_samples.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from make_figures import (C, _clean, _panel_title, _save, _suptitle,  # noqa: E402
                          FIGDIR, TASK_C)
from attn_convergence import (TASK_NAMES, js, loop_metrics, mass_tokens,  # noqa: E402
                              topk_overlap, _norm)

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import LogNorm                                 # noqa: E402

LOOP_C = ["#b9d0e4", "#7fa8c9", "#2c5f8a", "#8f5b2c"]


def _load_sample(npz):
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# 1. panorama — 96 rows, one per (loop, layer)                                #
# --------------------------------------------------------------------------- #
def fig_panorama(P, out, ncol=460):
    U, Ln, T = P.shape
    step = max(1, T // ncol)
    Tc = T // step
    M = P[:, :, :Tc * step].reshape(U, Ln, Tc, step).mean(-1)
    M = np.concatenate([M[u] for u in range(U)], 0)
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    fig.subplots_adjust(left=0.085, right=0.90, top=0.885, bottom=0.10)
    im = ax.imshow(M, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=max(M.min(), 1e-6), vmax=M.max()))
    for u in range(1, U):
        ax.axhline(u * Ln - 0.5, color="white", lw=1.0, ls="--", alpha=0.85)
    ax.set_yticks([Ln / 2 - 0.5 + u * Ln for u in range(U)])
    ax.set_yticklabels(["loop %d" % (u + 1) for u in range(U)])
    ax.set_xticks(np.linspace(0, Tc - 1, 6).astype(int))
    ax.set_xticklabels(["%d" % int(v * step) for v in
                        np.linspace(0, Tc - 1, 6)])
    ax.set_xlabel("key position in the prompt (token index, binned)")
    _panel_title(ax, "", "All 96 (loop, layer) attention maps of one sample")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(C["edge"]); s.set_linewidth(0.8)
    cax = fig.add_axes([0.912, 0.11, 0.013, 0.77])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("attention weight (log scale)", fontsize=8.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.8, length=2)
    _suptitle(fig, "The vertical stripe pattern repeats every 24 rows: "
                   "the layer determines where attention goes, the loop only "
                   "refines it", y=0.965)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 2. last-layer profile — four curves on top of each other                    #
# --------------------------------------------------------------------------- #
def fig_profile(P, out, layer=23):
    U, _, T = P.shape
    fig, ax = plt.subplots(figsize=(11.6, 3.5))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.145)
    for u in range(U):
        ax.plot(np.arange(T), P[u, layer], lw=1.1, color=LOOP_C[u], alpha=0.95,
                label="loop %d" % (u + 1))
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("key position (token index)")
    ax.set_ylabel("attention weight")
    _panel_title(ax, "", "Last layer (%d): attention of the final prompt token"
                 % layer)
    _clean(ax, grid="both")
    ax.legend(ncol=4, fontsize=8.6, loc="upper right")
    j = [js(P[u, layer], P[U - 1, layer]) for u in range(U)]
    _suptitle(fig, "Four loops, one layer — JS to the final loop: "
                   "%.4f / %.4f / %.4f  (loop 1 / 2 / 3)" %
              (j[0], j[1], j[2]), y=0.965)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 3. 96 x 96 JS distance matrix, layer-major                                  #
# --------------------------------------------------------------------------- #
def fig_dismat(P, out):
    U, Ln, _ = P.shape
    rows = [(i, u) for i in range(Ln) for u in range(U)]
    flat = np.array([_norm(P[u, i]) for i, u in rows])
    n = len(rows)
    D = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            v = js(flat[a], flat[b])
            D[a, b] = D[b, a] = v
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    fig.subplots_adjust(left=0.115, right=0.90, top=0.86, bottom=0.10)
    im = ax.imshow(D, cmap="viridis_r", vmin=0, vmax=min(D.max(), 0.4))
    for i in range(1, Ln):
        ax.axhline(i * U - .5, color="white", lw=.3, alpha=.35)
        ax.axvline(i * U - .5, color="white", lw=.3, alpha=.35)
    ax.set_xticks([0, n - 1])
    ax.set_xticklabels(["L0 · loop1", "L23 · loop4"], fontsize=8.4)
    ax.set_yticks([0, n - 1])
    ax.set_yticklabels(["L0 · loop1", "L23 · loop4"], fontsize=8.4)
    _panel_title(ax, "", "Pairwise JS distance, layer-major order")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(C["edge"]); s.set_linewidth(0.8)
    cax = fig.add_axes([0.912, 0.11, 0.016, 0.74])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("JS distance (nats)", fontsize=8.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.8, length=2)
    _suptitle(fig, "Each 4x4 block holds one layer across the four loops — "
                   "the dark diagonal blocks are the redundancy DynaLoop "
                   "exploits", y=0.955)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 4. one attention head, four loops                                           #
# --------------------------------------------------------------------------- #
def fig_heads(z, out, layer=23, head=2):
    ph = z["ph_last"]                       # [U, L, H, T]
    U = ph.shape[0]
    fig, axes = plt.subplots(1, U, figsize=(11.6, 2.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.155,
                        wspace=0.09)
    ref = ph[U - 1, layer, head].astype(np.float64)
    for u, ax in enumerate(axes):
        p = ph[u, layer, head].astype(np.float64)
        ax.plot(p, lw=0.9, color=LOOP_C[u] if u < len(LOOP_C) else C["dy"])
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel("key position")
        d = js(p, ref)
        _panel_title(ax, "", "loop %d" % (u + 1))
        ax.text(0.97, 0.94, "JS = %.4f" % d if u < U - 1 else "reference",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
                color=C["muted"])
        _clean(ax, grid=None)
    axes[0].set_ylabel("attention weight")
    _suptitle(fig, "A single head (layer %d, head %d): the distribution is "
                   "frozen after the second loop" % (layer, head), y=0.965)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 5. which tokens matter, and how early they are found                        #
# --------------------------------------------------------------------------- #
def fig_topk(z, P, out, top=14, layer=23, tokens=None):
    U, _, T = P.shape
    ref = P[U - 1, layer]
    idx = np.argsort(ref)[-top:][::-1]
    toks = tokens or []
    x = np.arange(top)
    fig, ax = plt.subplots(figsize=(11.6, 3.4))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.30)
    w = 0.2
    for u in range(U):
        ax.bar(x + (u - (U - 1) / 2) * w, P[u, layer][idx], width=w * 0.92,
               color=LOOP_C[u], lw=0, label="loop %d" % (u + 1))
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xticks(x)
    labels = []
    for i in idx:
        t = toks[i] if i < len(toks) else str(i)
        labels.append(t.replace("\u0120", "").replace("\u010a", "\\n"))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.8)
    ax.set_ylabel("attention weight")
    _panel_title(ax, "", "The %d keys the final loop attends to most" % top)
    _clean(ax, grid="y")
    ax.legend(ncol=4, fontsize=8.6, loc="upper right")
    _suptitle(fig, "Loops 2-4 already weight these keys almost identically; "
                   "loop 1 is still searching", y=0.965)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 6. sparsity — how few keys carry the mass                                   #
# --------------------------------------------------------------------------- #
def fig_sparsity(P, out, layer=23):
    U, _, T = P.shape
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.2))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.815, bottom=0.155,
                        wspace=0.26)
    ax = axes[0]
    for u in range(U):
        s = np.sort(P[u, layer])[::-1]
        ax.plot(np.arange(1, len(s) + 1), np.cumsum(s), lw=1.5,
                color=LOOP_C[u], label="loop %d" % (u + 1))
    ax.axhline(0.8, ls="--", lw=1.0, color=C["hot"])
    ax.set_xscale("log")
    ax.set_xlim(1, T)
    ax.set_xlabel("number of keys (sorted by weight)")
    ax.set_ylabel("cumulative attention mass")
    _panel_title(ax, "a", "Cumulative mass")
    _clean(ax, grid="both")
    ax.legend(fontsize=8.4, loc="lower right")

    ax = axes[1]
    frac = [0.5, 0.8, 0.9, 0.95]
    for u in range(U):
        s = np.sort(P[u, layer])[::-1]
        c = np.cumsum(s)
        cnt = [int(np.searchsorted(c, f) + 1) for f in frac]
        ax.plot([f * 100 for f in frac], [100.0 * v / T for v in cnt], "o-",
                lw=1.6, ms=4.5, color=LOOP_C[u],
                markeredgecolor="white", markeredgewidth=1.0,
                label="loop %d" % (u + 1))
    ax.set_xlabel("attention mass (%)")
    ax.set_ylabel("keys required (% of prompt)")
    _panel_title(ax, "b", "Keys required per mass level")
    _clean(ax, grid="both")
    n80 = mass_tokens(P[U - 1, layer], 0.8)
    ax.annotate("80%% mass: %d keys\n(%.2f%% of the prompt)"
                % (n80, 100.0 * n80 / T), xy=(80, 100.0 * n80 / T),
                xycoords="data", textcoords="axes fraction",
                xytext=(0.06, 0.82), fontsize=8.2, color=C["hot"],
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C["hot"],
                                connectionstyle="arc3,rad=-0.15"))
    _suptitle(fig, "Attention is extremely sparse: a few dozen keys carry "
                   "almost all of the mass", y=0.965)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# 7. convergence speed is a property of the task                              #
# --------------------------------------------------------------------------- #
def fig_task_convergence(buckets, out):
    tasks = [t for t in ("NIAH-Multi", "GSM8K", "MATH-500") if t in buckets]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.835, bottom=0.155)
    for t in tasks:
        dl = np.asarray([np.asarray(s["js_adjacent_loop"])
                         for s in buckets[t]])
        m = dl.mean(axis=0)
        ax.plot([1, 2, 3], m, "o-", lw=1.9, ms=5.4, color=TASK_C.get(t, C["dy"]),
                markeredgecolor="white", markeredgewidth=1.1, label=t)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("loop transition  $t \\to t+1$   (point = $d_t$)")
    ax.set_ylabel("JS displacement  $d_t$  (nats)")
    ax.axhline(0.01, ls="--", lw=1.0, color=C["faint"])
    ax.text(2.98, 0.0107, "$d = 0.01$", fontsize=8.0, ha="right",
            color=C["muted"], va="bottom")
    _panel_title(ax, "", "Inter-loop displacement by task")
    _clean(ax, grid="both")
    ax.legend(fontsize=8.6, loc="lower left")
    _suptitle(fig, "Retrieval keeps redistributing attention into loop 3; "
                   "reasoning has already frozen by loop 2", y=0.965)
    return _save(fig, out)


# --------------------------------------------------------------------------- #
def _per_sample_rows(dirs):
    buckets, rows = {}, []
    for d in dirs:
        for npz in sorted(glob.glob(os.path.join(d, "*_attn.npz"))):
            meta_p = npz.replace("_attn.npz", "_meta.json")
            if not os.path.exists(meta_p):
                continue
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            task = TASK_NAMES.get(str(meta.get("task", "")).lower(), "?")
            with np.load(npz) as z:
                P = z["p_last"].astype(np.float32)
            m = loop_metrics(P)
            buckets.setdefault(task, []).append(m)
            dl = np.asarray(m["js_adjacent_loop"])
            rows.append({
                "task": task,
                "sample": os.path.basename(npz).replace("_attn.npz", ""),
                "T": m["n_tokens"],
                "j1": m["js_vs_final"][0], "j2": m["js_vs_final"][1],
                "j3": m["js_vs_final"][2],
                "ov3": m["top10_overlap"][2],
                "n80": m["mass80_tokens"],
                "d_loop": float(dl.mean()),
                "d_layer": m["js_adjacent_layer_mean"],
            })
    return buckets, rows


def _write_table(rows, path):
    order = {"NIAH-Multi": 0, "GSM8K": 1, "MATH-500": 2}
    rows.sort(key=lambda r: (order.get(r["task"], 9), r["sample"]))
    out = ["| 样本 | tokens | JS(loop1, loop4) | JS(loop2, loop4) | "
           "JS(loop3, loop4) | loop3 与 loop4 的 top-10% 重合 | "
           "80% 质量所需 key 数 | 跨循环漂移 $\\bar d$ | 同循环跨层漂移 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| %s | %d | %.4f | %.4f | %.4f | %.1f%% | %d | %.4f | %.4f |"
                   % (r["sample"], r["T"], r["j1"], r["j2"], r["j3"],
                      r["ov3"] * 100, r["n80"], r["d_loop"], r["d_layer"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-dir", action="append", default=None)
    ap.add_argument("--main", default="math500_d0",
                    help="sample tag used for the single-sample figures")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--out-dir", default=FIGDIR)
    args = ap.parse_args()

    dirs = args.attn_dir or [os.path.join(ROOT, "results", "attn")]
    dirs = [d for d in dirs if os.path.isdir(d)]

    main_npz = None
    for d in dirs:
        cand = os.path.join(d, args.main + "_attn.npz")
        if os.path.exists(cand):
            main_npz = cand
            break
    if main_npz is None:
        print("sample %s not found in %s" % (args.main, dirs), file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    z = _load_sample(main_npz)
    P = z["p_last"].astype(np.float64)
    meta_p = main_npz.replace("_attn.npz", "_meta.json")
    meta = {}
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    print("main sample:", main_npz)

    fig_panorama(P, os.path.join(args.out_dir, "attn_panorama.png"))
    fig_profile(P, os.path.join(args.out_dir, "attn_profile.png"),
                layer=args.layer)
    fig_dismat(P, os.path.join(args.out_dir, "attn_dismat.png"))
    if "ph_last" in z:
        fig_heads(z, os.path.join(args.out_dir, "attn_heads.png"),
                  layer=args.layer)
    fig_topk(z, P, os.path.join(args.out_dir, "attn_topk.png"),
             layer=args.layer, tokens=meta.get("tokens"))
    fig_sparsity(P, os.path.join(args.out_dir, "attn_sparsity.png"),
                 layer=args.layer)

    buckets, rows = _per_sample_rows(dirs)
    fig_task_convergence(buckets,
                         os.path.join(args.out_dir, "attn_task_convergence.png"))
    p = _write_table(rows, os.path.join(ROOT, "results", "attn_samples.md"))
    print("table ->", p, "(%d samples)" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
