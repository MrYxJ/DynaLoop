"""Regenerate every figure in ``docs/figures/`` from the JSON in ``results/``.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py --only 0 3

Figures whose source JSON is missing are skipped with a warning, so the script
can be run on a fresh clone (the two schematic figures, 1 and 2, need no data).
Sources::

    results/attn_convergence.json   <- scripts/attn_convergence.py  (fig 0)
    results/budget_heatmap.json     <- examples/budget_heatmap.py   (fig 3, 4)
    results/len_scaling_s21.json    <- scripts/run_len_scaling.py   (fig 5)
    results/lmeval_*.json           <- scripts/run_math.py          (fig 6)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "results")

# --------------------------------------------------------------------------- #
# visual identity                                                             #
# --------------------------------------------------------------------------- #
C = {
    "ink": "#16202e",        # primary text
    "muted": "#6b7688",      # secondary text / tick labels
    "faint": "#aab2c0",
    "grid": "#e7ebf1",
    "edge": "#c8d0dc",
    "dy": "#2c5f8a",         # DynaLoop
    "dy_l": "#b9d0e4",
    "full": "#8f9bb0",       # FullKV
    "dkv": "#c98a2b",        # DynamicKV
    "hot": "#a83e32",        # emphasis / floor
    "good": "#2f7d63",
}
TASK_C = {"NIAH-Multi": "#0f7c74", "GSM8K": C["dy"], "MATH-500": "#c1622f"}
TASK_ORDER = ["NIAH-Multi", "GSM8K", "MATH-500"]

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch)    # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": C["edge"],
    "axes.labelcolor": C["ink"],
    "axes.titlecolor": C["ink"],
    "axes.titlelocation": "left",
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.9,
    "axes.labelpad": 5,
    "text.color": C["ink"],
    "xtick.color": C["muted"],
    "ytick.color": C["muted"],
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 8.6,
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name, dpi=200):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  ->", os.path.relpath(p, ROOT))
    return p


def _clean(ax, grid="y"):
    """Despine, soften the grid, and put grid lines behind the data."""
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(C["edge"])
    ax.spines["bottom"].set_color(C["edge"])
    if grid:
        ax.grid(axis=grid, alpha=0.75, lw=0.7, color=C["grid"])
    ax.tick_params(length=2.6, width=0.8, pad=2.5)


def _tag(ax, t):
    ax.text(-0.015, 1.05, t, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", color=C["ink"], va="bottom", ha="right")


def _suptitle(fig, text, y=0.985):
    fig.suptitle(text, fontsize=11.8, fontweight="bold", color=C["ink"],
                 y=y, ha="left", x=0.012)
    return fig


def _panel_title(ax, tag, text):
    ax.set_title(text, fontsize=10.2, fontweight="bold", pad=6, loc="center")
    ax.text(-0.02, 1.03, tag, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=C["faint"], va="bottom", ha="right")


def _mean(v):
    a = np.asarray(v, dtype=np.float64)
    return (a.mean(axis=0).tolist() if a.ndim > 1 else float(a.mean()))


def _std(v):
    a = np.asarray(v, dtype=np.float64)
    return (a.std(axis=0).tolist() if a.ndim > 1 else float(a.std()))


# --------------------------------------------------------------------------- #
# Figure 0 — the empirical foundation: attention converges across loops        #
# --------------------------------------------------------------------------- #
def fig_attention_convergence():
    """Why a loop-aware budget is possible at all (measured, 25 samples)."""
    d = _load("attn_convergence.json")
    if not d:
        print("  !! results/attn_convergence.json missing, skipping fig0 "
              "(run scripts/attn_convergence.py)")
        return None
    tasks = [t for t in TASK_ORDER if t in d["tasks"]] or list(d["tasks"])
    if not tasks:
        print("  !! empty convergence data, skipping fig0")
        return None

    loops = np.arange(1, 5)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.885, bottom=0.085,
                        wspace=0.26, hspace=0.42)

    # ---- (a) distance to the final loop -------------------------------- #
    ax = axes[0][0]
    for t in tasks:
        v = d["tasks"][t]
        m = np.asarray(v["js_vs_final"]["mean"][:3])
        s = np.asarray(v["js_vs_final"]["std"][:3])
        ax.plot(loops[:3], m, "o-", lw=1.9, ms=5.4, color=TASK_C.get(t, C["dy"]),
                markeredgecolor="white", markeredgewidth=1.1, label=t)
        ax.fill_between(loops[:3], m - s, np.clip(m + s, 1e-4, None),
                        color=TASK_C.get(t, C["dy"]), alpha=0.13, lw=0)
    ref = np.mean([d["tasks"][t]["js_layer0_last"]["mean"] for t in tasks])
    ax.axhline(ref, ls="--", lw=1.1, color=C["full"])
    ax.text(2.97, ref * 0.72, "same loop, layer 0 vs. 23", fontsize=7.8,
            ha="right", color=C["muted"])
    ax.axhspan(1e-4, 3e-3, color=C["good"], alpha=0.08, lw=0)
    ax.text(1.02, 4e-4, "converged", fontsize=7.8, color=C["good"])
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 1.4)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("loop index  $t$")
    ax.set_ylabel("JS to the final loop  (nats)")
    _panel_title(ax, "a", "Attention converges after the first loop")
    _clean(ax, grid="both")
    ax.legend(loc="upper right", fontsize=8.2)

    # ---- (b) which tokens are attended ---------------------------------- #
    ax = axes[0][1]
    for t in tasks:
        v = d["tasks"][t]
        m = np.asarray(v["top10_overlap"]["mean"])
        s = np.asarray(v["top10_overlap"]["std"])
        ax.plot(loops, m, "o-", lw=1.9, ms=5.4, color=TASK_C.get(t, C["dy"]),
                markeredgecolor="white", markeredgewidth=1.1, label=t)
        ax.fill_between(loops, m - s, np.clip(m + s, 0, 1),
                        color=TASK_C.get(t, C["dy"]), alpha=0.13, lw=0)
    ax.set_ylim(0.45, 1.05)
    ax.set_xticks(loops)
    ax.set_xlabel("loop index  $t$")
    ax.set_ylabel("overlap with the final top-10% keys")
    _panel_title(ax, "b", "The important-token set is locked early")
    _clean(ax, grid="y")

    # ---- (c) cross-loop vs cross-layer displacement --------------------- #
    ax = axes[1][0]
    w = 0.2
    for j, t in enumerate(tasks):
        v = d["tasks"][t]
        dl = np.asarray(v["js_adjacent_loop"]["mean"])
        ll = float(v["js_adjacent_layer_mean"]["mean"])
        col = TASK_C.get(t, C["dy"])
        x = np.arange(3) + (j - (len(tasks) - 1) / 2) * w
        ax.bar(x, dl, width=w * 0.92, color=col, label=t if j == 0 else None)
        ax.plot([x[0] - w * 0.5, x[-1] + w * 0.5], [ll, ll], ls="--", lw=1.0,
                color=col, alpha=0.9)
        if j == len(tasks) - 1:
            for xi, yi in zip(x, dl):
                pass
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["$d_1$\n1$\\to$2", "$d_2$\n2$\\to$3", "$d_3$\n3$\\to$4"],
                       fontsize=8.4)
    ax.set_ylabel("JS displacement  (nats)")
    ax.text(0.50, 0.945, "dashed: adjacent layers inside one loop",
            transform=ax.transAxes, fontsize=7.6, ha="center", va="top",
            color=C["muted"])
    _panel_title(ax, "c", "Cross-loop drift falls below cross-layer drift")
    _clean(ax, grid="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TASK_C.get(t, C["dy"]))
               for t in tasks]
    ax.legend(handles, tasks, fontsize=8.2, loc="upper right", ncol=1)

    # ---- (d) convergence speed is a task property ----------------------- #
    ax = axes[1][1]
    for t in tasks:
        v = d["tasks"][t]
        dl = np.asarray(v["js_adjacent_loop"]["mean"])
        rho = dl[1:] / np.maximum(dl[:-1], 1e-12)
        col = TASK_C.get(t, C["dy"])
        ax.plot([2, 3], rho, "o-", lw=1.9, ms=5.4, color=col,
                markeredgecolor="white", markeredgewidth=1.1,
                label="%s  (%.2f, %.2f)" % (t, rho[0], rho[1]))
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 6e-1)
    ax.set_xticks([2, 3])
    ax.set_xlabel("loop transition  $t \\to t+1$")
    ax.set_ylabel("$\\rho_t = d_t / d_{t-1}$")
    _panel_title(ax, "d", "Convergence speed is task-dependent")
    _clean(ax, grid="both")
    ax.legend(fontsize=8.2, loc="lower left")

    _suptitle(fig, "Empirical foundation: within a layer, attention "
                   "redistributes once and then freezes "
                   "(Ouro-1.4B, 4 loops $\\times$ 24 layers)", y=0.985)
    return _save(fig, "fig0_attention_convergence.png")


# --------------------------------------------------------------------------- #
# Figure 1 — what the method is (no data needed)                              #
# --------------------------------------------------------------------------- #
def fig_overview():
    """The looped backbone, the 96 KV units, and how one unit's budget is set."""
    fig = plt.figure(figsize=(13.6, 4.5))
    fig.subplots_adjust(left=0.018, right=0.985, top=0.90, bottom=0.02)

    # ---------------- panel a : the looped backbone ----------------------
    ax = fig.add_axes([0.02, 0.065, 0.225, 0.755]); ax.axis("off")
    _panel_title(ax, "a", "Looped LM: one weight stack, executed $T$ times")
    for t in range(4):
        y = 0.80 - t * 0.205
        fc = C["dy_l"] if t == 0 else "#eef3f9"
        ax.add_patch(FancyBboxPatch((0.30, y - 0.058), 0.66, 0.126,
                                    boxstyle="round,pad=0.010,rounding_size=0.030",
                                    fc=fc, ec=C["dy"] if t == 0 else C["edge"],
                                    lw=1.25))
        ax.text(0.63, y + 0.004, f"loop {t + 1}", ha="center", va="center",
                fontsize=9.4, fontweight="bold" if t == 0 else "normal",
                color=C["dy"] if t == 0 else C["ink"])
        ax.text(0.63, y - 0.040, "24 shared layers", ha="center", va="center",
                fontsize=7.8, color=C["muted"])
    ax.annotate("", xy=(0.225, 0.795), xytext=(0.225, 0.115),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C["hot"],
                                shrinkA=2, shrinkB=2))
    ax.text(0.155, 0.455, "same weights", ha="center", va="center",
            fontsize=8.2, color=C["hot"], rotation=90)
    ax.text(0.63, 0.015, "$T{\\times}L$ $(loop,\\,layer)$ units  =  "
                        "4 ${\\times}$ 24 = 96  (Ouro-1.4B)",
            ha="center", va="bottom", fontsize=8.6, color=C["muted"])

    # ---------------- panel b : the per-unit grid ------------------------
    ax = fig.add_axes([0.285, 0.155, 0.365, 0.63])
    _panel_title(ax, "b", "Each unit receives its own budget")
    rng = np.random.default_rng(7)
    base = np.array([1.0, 0.53, 0.15, 0.13])
    m = np.clip(base[:, None] + rng.normal(0, 0.055, (4, 24)), 0.05, 1.0)
    m[0, :] = 1.0
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([0, 5, 10, 15, 20, 23])
    ax.set_xticklabels(["0", "5", "10", "15", "20", "23"])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"loop {i + 1}" for i in range(4)])
    ax.set_xlabel("layer  $\\ell$")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(C["edge"]); s.set_linewidth(0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.022)
    cb.set_label("KV retention ratio", fontsize=8.4)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.8, length=2)
    ax.text(0.015, 0.055, "loop 1: uncompressed (calibration reference)",
            transform=ax.transAxes, fontsize=7.6, color=C["muted"],
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))

    # ---------------- panel c : the decision chain -----------------------
    ax = fig.add_axes([0.700, 0.055, 0.292, 0.765]); ax.axis("off")
    _panel_title(ax, "c", "Budget from measured inter-loop displacement")
    steps = [
        ("same layer, one loop apart",
         "$s_{t-1,\\ell}$ ,  $s_{t,\\ell}$"),
        ("how far did this layer still move?",
         "$d = \\mathrm{JS}(s_{t-1,\\ell}\\,\\Vert\\,s_{t,\\ell})$"),
        ("normalised by loop-1 motion (self-calibration)",
         "$\\rho = d / D_1$"),
        ("one rule, three knobs",
         "$r = \\mathrm{clip}(\\kappa\\rho^{\\gamma},\\,f,\\,1)$"),
        ("ChunkKV then selects which tokens",
         "$B = r\\,(S - w)$"),
    ]
    y = 0.845
    for i, (txt, eq) in enumerate(steps):
        col = [C["faint"], C["dy"], C["dy"], C["hot"], C["good"]][i]
        ax.add_patch(FancyBboxPatch((0.015, y - 0.048), 0.97, 0.122,
                                    boxstyle="round,pad=0.006,rounding_size=0.022",
                                    fc="white", ec=col, lw=1.15))
        ax.text(0.055, y + 0.028, eq, fontsize=10.2, va="center", color=C["ink"])
        ax.text(0.055, y - 0.024, txt, fontsize=7.9, va="center",
                color=C["muted"])
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.094), xytext=(0.5, y - 0.052),
                        arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                        color=C["faint"]))
        y -= 0.172

    _suptitle(fig, "DynaLoop: a per-(loop, layer) KV budget derived from "
                   "measured attention displacement", y=0.965)
    return _save(fig, "fig1_overview.png")


# --------------------------------------------------------------------------- #
# Figure 2 — the allocation rule                                              #
# --------------------------------------------------------------------------- #
def fig_rule():
    r"""$r = clip(\kappa \rho^\gamma, f, 1)$ — the three knobs made visible."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), sharey=True)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.795, bottom=0.135,
                        wspace=0.15)
    rho = np.linspace(0, 1.4, 500)
    floor = 0.10

    def curve(k, g, f=floor):
        return np.clip(k * np.power(rho, g), f, 1.0)

    # ---- a: kappa ----
    ax = axes[0]
    for k in (0.25, 0.5, 0.75, 1.0):
        ax.plot(rho, curve(k, 1.0), lw=2.0, color=C["dy"],
                alpha=0.28 + 0.62 * k,
                label="$\\kappa$ = %.2f" % k)
    ax.set_title("$\\kappa$: budget of the first compressed loop",
                 fontsize=10.2, loc="center")
    ax.set_xlabel("$\\rho$  —  movement relative to loop 1")

    # ---- b: gamma ----
    ax = axes[1]
    for g, a in zip((0.5, 1.0, 2.0, 4.0), (0.35, 0.55, 0.78, 1.0)):
        ax.plot(rho, curve(0.5, g), lw=2.0, color=C["dy"], alpha=a,
                label="$\\gamma$ = %.1f" % g)
    ax.set_title("$\\gamma$: how fast later loops collapse", fontsize=10.2,
                 loc="center")
    ax.set_xlabel("$\\rho$")

    # ---- c: floor ----
    ax = axes[2]
    for f, a in zip((0.0, 0.05, 0.10, 0.20), (0.35, 0.55, 0.78, 1.0)):
        ax.plot(rho, curve(0.5, 1.0, f), lw=2.0, color=C["dy"], alpha=a,
                label="$f$ = %.2f" % f)
    ax.set_title("$f$: floor — never evict below this", fontsize=10.2,
                 loc="center")
    ax.set_xlabel("$\\rho$")

    for ax in axes:
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(0, 1.35)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
        ax.axhline(floor, ls="--", lw=1.0, color=C["hot"], alpha=0.85)
        ax.axhline(1.0, ls=":", lw=1.0, color=C["faint"])
        ax.fill_between(rho, 0, floor, color=C["hot"], alpha=0.05, lw=0)
        ax.legend(loc="lower right", fontsize=8.4)
        _clean(ax, grid="both")
    axes[0].set_ylabel("KV retention ratio  $r$")
    _panel_title(axes[0], "a", "$\\kappa$: budget of the first compressed loop")
    _panel_title(axes[1], "b", "$\\gamma$: speed of collapse")
    _panel_title(axes[2], "c", "$f$: retention floor")
    _suptitle(fig, "One rule governs all 96 units:   "
                   "$r=\\mathrm{clip}(\\kappa\\rho^{\\gamma},\\,f,\\,1)$",
              y=0.965)
    return _save(fig, "fig2_budget_rule.png")


# --------------------------------------------------------------------------- #
# helpers for measured figures                                                #
# --------------------------------------------------------------------------- #
def _lmeval(tag_glob):
    hits = sorted(glob.glob(os.path.join(OUT, tag_glob)))
    if not hits:
        return None
    with open(hits[-1], encoding="utf-8") as f:
        return json.load(f)


def _acc(d):
    v = list(d["results"].values())[0]["exact_match,none"]
    return None if v in ("N/A", None) else float(v)


_HEATMAP_TASKS = [("NIAH-Multi (retrieval)", "NIAH-Multi"),
                  ("GSM8K (reasoning)", "GSM8K"),
                  ("MATH-500 (reasoning)", "MATH-500")]


# --------------------------------------------------------------------------- #
# Figure 3 — adaptive vs uniform across loops                                 #
# --------------------------------------------------------------------------- #
def fig_per_loop():
    """Per-loop retention measured by ``examples/budget_heatmap.py``."""
    rep = _load("budget_heatmap.json")
    if not rep:
        print("  !! results/budget_heatmap.json missing, skipping fig3")
        return None
    data = {}
    for key, short in _HEATMAP_TASKS:
        if key in rep["tasks"]:
            m = np.asarray(rep["tasks"][key]["per_unit_retention"])
            data[short] = m.mean(axis=1).tolist()
    if not data:
        print("  !! no per-loop data, skipping fig3")
        return None

    names = [t for t in TASK_ORDER if t in data] or list(data)
    fig, axes = plt.subplots(1, len(names), figsize=(3.55 * len(names) + 0.7,
                                                     3.35), sharey=True)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.80, bottom=0.145,
                        wspace=0.13)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        vals = np.asarray(data[name])
        x = np.arange(4)
        tot = float(vals.mean())
        ax.bar(x - 0.195, vals, width=0.37, color=C["dy"], lw=0,
               label="DynaLoop")
        ax.bar(x + 0.195, [tot] * 4, width=0.37, color=C["dkv"], lw=0,
               alpha=0.55, label="uniform (%.2f)" % tot)
        for i, v in enumerate(vals):
            ax.text(i - 0.195, v + 0.025, "%.2f" % v, ha="center",
                    fontsize=7.9, color=C["dy"])
        ax.set_xticks(x)
        ax.set_xticklabels(["loop %d" % (i + 1) for i in x])
        ax.set_ylim(0, 1.20)
        _panel_title(ax, "", name)
        _clean(ax, grid="y")
    axes[0].set_ylabel("fraction of prompt tokens retained")
    axes[0].legend(loc="upper right", fontsize=8.2)
    fig.text(0.5, 0.028, "the same 24 layers, seen four times",
             fontsize=8.8, color=C["muted"], ha="center")
    _suptitle(fig, "Allocated retention per loop, against a uniform schedule "
                   "of equal total budget", y=0.965)
    return _save(fig, "fig3_per_loop.png")


# --------------------------------------------------------------------------- #
# Figure 4 — the measured per-(loop, layer) heatmap                           #
# --------------------------------------------------------------------------- #
def fig_heatmap():
    rep = _load("budget_heatmap.json")
    if not rep:
        print("  !! results/budget_heatmap.json missing, skipping fig4")
        return None
    tasks = [t for t in rep["tasks"]
             if t in ("NIAH-Multi (retrieval)", "GSM8K (reasoning)",
                      "MATH-500 (reasoning)")]
    if not tasks:
        tasks = list(rep["tasks"])
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(3.62 * n + 0.75, 3.75))
    fig.subplots_adjust(left=0.055, right=0.905, top=0.775, bottom=0.13,
                        wspace=0.30)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, tasks):
        m = np.asarray(rep["tasks"][name]["per_unit_retention"])
        im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues", aspect="auto",
                       interpolation="nearest")
        _panel_title(ax, "", name)
        ax.set_xlabel("layer  $\\ell$", fontsize=9)
        ax.set_xticks([0, 5, 10, 15, 20, 23])
        ax.set_xticklabels(["0", "5", "10", "15", "20", "23"])
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(["loop %d" % (i + 1) for i in range(m.shape[0])])
        ax.text(0.5, -0.30, "mean retention  %.3f"
                % rep["tasks"][name]["mean_retention"],
                transform=ax.transAxes, ha="center", fontsize=8.4,
                color=C["muted"])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(C["edge"]); s.set_linewidth(0.8)
    cax = fig.add_axes([0.922, 0.155, 0.014, 0.60])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("KV retention ratio", fontsize=8.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.8, length=2)
    _suptitle(fig, "Measured retention of every (loop, layer) unit "
                   "(Ouro-1.4B, $\\kappa$=0.5, $\\gamma$=1.0, $f$=0.10)",
              y=0.95)
    return _save(fig, "fig4_heatmap.png")


# --------------------------------------------------------------------------- #
# Figure 5 — FlashLoop-style length scaling                                   #
# --------------------------------------------------------------------------- #
def fig_len_scaling():
    d = _load("len_scaling_s21.json")
    if not d:
        print("  !! len_scaling json missing, skipping fig5")
        return None
    cells = d["cells"]
    lens = sorted(int(k) for k in cells)
    ok = [L for L in lens if cells[str(L)]["fullkv"].get("status") == "ok"]
    oom = [L for L in lens if L not in ok]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.35))
    fig.subplots_adjust(left=0.062, right=0.985, top=0.795, bottom=0.145,
                        wspace=0.22)

    def pair(ax, key, ylabel, title, tag):
        yf = [cells[str(L)]["fullkv"][key] for L in ok]
        yd = [cells[str(L)]["dynaloop"][key] for L in ok]
        ax.plot(ok, yf, "o-", color=C["full"], lw=1.9, ms=5.2,
                markeredgecolor="white", markeredgewidth=1.0, label="FullKV")
        ax.plot(ok, yd, "o-", color=C["dy"], lw=1.9, ms=5.2,
                markeredgecolor="white", markeredgewidth=1.0, label="DynaLoop")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens)
        ax.set_xticklabels([str(L) for L in lens])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(ylabel)
        _panel_title(ax, tag, title)
        _clean(ax, grid="both")
        ax.legend(loc="upper left", fontsize=8.4)
        return yf, yd

    ax = axes[0]
    yf, yd = pair(ax, "kv_MiB", "KV cache (MiB)", "KV cache memory", "a")
    if oom:
        x0, b = ok[-1], cells[str(ok[-1])]
        xs = [x0] + oom
        ax.plot(xs, [b["fullkv"]["kv_MiB"] * (L / x0) for L in xs], "--",
                color=C["full"], lw=1.3)
        ax.plot(xs, [b["dynaloop"]["kv_MiB"] * (L / x0) for L in xs], "--",
                color=C["dy"], lw=1.3)
        ax.plot(oom, [b["fullkv"]["kv_MiB"] * (L / x0) for L in oom], "s",
                color=C["full"], ms=4.0, mfc="white", lw=1.1)
        ax.plot(oom, [b["dynaloop"]["kv_MiB"] * (L / x0) for L in oom], "s",
                color=C["dy"], ms=4.0, mfc="white", lw=1.1)
        ax.set_ylim(0, b["fullkv"]["kv_MiB"] * (oom[-1] / x0) * 1.22)
        ax.annotate("extrapolated — beyond 4K\nOOMs on an 8 GB GPU",
                    xy=(oom[0], b["dynaloop"]["kv_MiB"] * (oom[0] / x0)),
                    xytext=(oom[0] * 0.92,
                            b["fullkv"]["kv_MiB"] * (oom[0] / x0) * 0.30),
                    fontsize=7.5, color=C["muted"], va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=C["faint"]))
    else:
        for L, a, b in zip(ok, yf, yd):
            ax.annotate("%+.0f%%" % ((b / a - 1) * 100), (L, b),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.8, color=C["dy"])

    pair(axes[1], "prefill_s", "prefill latency (s)", "Prefill latency", "b")
    pair(axes[2], "decode_ms", "per-step decode (ms)", "Decoding latency", "c")
    if oom:  # mark the unmeasured region in the two latency panels
        for ax in (axes[1], axes[2]):
            x0 = ok[-1]
            ax.axvspan(x0 * 1.18, lens[-1] * 1.06, color=C["grid"],
                       alpha=0.55, lw=0, zorder=0)
            ax.text(x0 * 2.4, 0.06, "not measured\n(OOM)", fontsize=7.4,
                    color=C["muted"], va="bottom", ha="left",
                    transform=ax.get_xaxis_transform())

    _suptitle(fig, "Scaling with context length: DynaLoop vs. FullKV "
                   "(Ouro-1.4B, $\\kappa$=0.5, $\\gamma$=1.0, "
                   "RTX 4070 Laptop 8 GB)", y=0.965)
    return _save(fig, "fig5_len_scaling.png")


# --------------------------------------------------------------------------- #
# Figure 6 — accuracy / memory trade-off                                      #
# --------------------------------------------------------------------------- #
def fig_accuracy_kv():
    groups = [
        ("GSM8K", [("FullKV", "*full_gsm8k.json", C["full"]),
                   ("DynamicKV", "*dkv_gsm8k.json", C["dkv"]),
                   ("DynaLoop", "*k50_gsm8k.json", C["dy"])]),
        ("MATH-500", [("FullKV", "*full_math500.json", C["full"]),
                      ("DynamicKV", "*dkv_math500.json", C["dkv"]),
                      ("DynaLoop", "*k50_math500.json", C["dy"])]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    fig.subplots_adjust(left=0.078, right=0.975, top=0.795, bottom=0.145,
                        wspace=0.24)
    for ax, (title, items) in zip(axes, groups):
        pts = []
        for label, g, col in items:
            d = _lmeval(g)
            if not d:
                continue
            a, kv = _acc(d), d.get("kv_MiB_per_sample")
            if a is None or not kv:
                continue
            pts.append((float(kv), float(a), col, label))
        if not pts:
            ax.axis("off")
            ax.set_title(title + "\n(pending)")
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", lw=1.0, color=C["faint"], zorder=1)
        for x, y, c, l in pts:
            ax.scatter([x], [y], s=115, color=c, zorder=3, edgecolor="white",
                       linewidth=1.5)
            off = {"FullKV": (0, -34), "DynamicKV": (0, 13),
                   "DynaLoop": (0, -34)}.get(l, (0, -34))
            kw = (dict(bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
                  if l == "DynamicKV" else {})
            ax.annotate("%s\n%.1f%% · %.0f MiB" % (l, y * 100, x), (x, y),
                        textcoords="offset points", xytext=off,
                        ha="center", fontsize=8.1, color=c, zorder=4, **kw)
        top = max(ys)
        ax.axhline(top, ls=":", lw=1.0, color=C["hot"], alpha=0.8, zorder=0)
        ax.text(max(xs) + 0.20 * (max(xs) - min(xs)), top + 0.008,
                "FullKV level", fontsize=7.6, ha="right", color=C["hot"])
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - 0.30 * span, max(xs) + 0.24 * span)
        ax.set_ylim(min(ys) - 0.11, top + 0.14)
        ax.set_xlabel("KV cache per sample (MiB)")
        ax.set_title(title, fontsize=10.2, loc="center")
        _clean(ax, grid="y")
    axes[0].set_ylabel("exact-match accuracy (n = 100)")
    _suptitle(fig, "Accuracy–memory trade-off", y=0.965)
    return _save(fig, "fig6_accuracy_vs_kv.png")


FIGFNS = {0: fig_attention_convergence, 1: fig_overview, 2: fig_rule,
          3: fig_per_loop, 4: fig_heatmap, 5: fig_len_scaling,
          6: fig_accuracy_kv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    keys = args.only or sorted(FIGFNS)
    for k in keys:
        print("fig%d" % k)
        FIGFNS[k]()


if __name__ == "__main__":
    main()
