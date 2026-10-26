"""Dump + plot the realised per-(loop, layer) KV budget.

This is the script behind ``docs/figures/budget_heatmap.png``.  It shows the
thing DynaLoop is *about*: the budget is decided per **(loop, layer)** unit,
not per layer and not per loop, and the resulting pattern differs between
tasks because it is measured from the attention, not written down.

    python examples/budget_heatmap.py --out docs/figures/budget_heatmap.png

Outputs
-------
``<out>.png``   heatmap grid, one panel per task
``<out>.json``  the raw retention matrix, so the figure can be regenerated
                without a GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynaloop import BatchedOuroEngine, DynaLoopConfig, load_ouro  # noqa: E402
from dynaloop.eval.datasets import (GSM8K_FEWSHOT_8, MATH_FEWSHOT_8,  # noqa: E402
                                    build_chat_prompt, eval_indices,
                                    load_gsm8k, load_math500)
from dynaloop.eval.niah import build_niah_multi, encode, niah_prompt  # noqa: E402


def build_prompts(tok, args):
    """One batch of prompts per task, all about the same context length."""
    out = {}

    # --- retrieval-style: multi-key needle-in-a-haystack ---------------
    items = [build_niah_multi(args.ctx, args.needles, seed=5000 + i)
             for i in range(args.per_task)]
    out["NIAH-Multi (retrieval)"] = [niah_prompt(it) for it in items]

    # --- reasoning-style: GSM8K / MATH-500 few-shot chat prompts --------
    gsm = load_gsm8k()
    math = load_math500()
    gi = eval_indices(len(gsm), args.per_task, 42)
    mi = eval_indices(len(math), args.per_task, 42)
    gsm_pairs = [(gsm[i]["question"],
                  gsm[i]["solution"].split("####")[-1].strip()) for i in gi]
    math_pairs = [(math[i]["problem"], math[i]["solution"]) for i in mi]
    out["GSM8K (reasoning)"] = [
        build_chat_prompt(tok, q, task="gsm8k", fewshot=GSM8K_FEWSHOT_8)
        for q, _ in gsm_pairs]
    out["MATH-500 (reasoning)"] = [
        build_chat_prompt(tok, q, task="math", fewshot=MATH_FEWSHOT_8)
        for q, _ in math_pairs]
    return out


def measure(model, tok, cfg, prompts, device):
    """Mean per-(loop, layer) retention over the given prompts."""
    enc = [encode(tok, p)[0] for p in prompts]
    S = max(len(x) for x in enc)
    ids = torch.full((len(enc), S), tok.pad_token_id or 0, dtype=torch.long,
                     device=device)
    for j, x in enumerate(enc):
        ids[j, :len(x)] = torch.tensor(x, device=device)
    lens = torch.tensor([len(x) for x in enc], device=device)

    eng = BatchedOuroEngine(model, tok, cfg, device=device)
    eng.prefill(ids, lens)
    H, M = eng.H, eng.M
    mat = torch.zeros(H, M)
    for (ut, layer), t in eng._prompt_fill.items():
        mat[ut, layer] = float((t.float() / lens.float()).mean())
    ret = float(mat.mean())
    kv = int(eng.kv_bytes() / max(len(enc), 1))
    del eng
    torch.cuda.empty_cache()
    return mat.tolist(), ret, kv, int(lens.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DYNALOOP_MODEL",
                                                      "Ouro-1.4B"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--needles", type=int, default=16)
    ap.add_argument("--per-task", type=int, default=4)
    ap.add_argument("--js-kappa", type=float, default=0.5)
    ap.add_argument("--js-gamma", type=float, default=1.0)
    ap.add_argument("--js-floor", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures",
                                                  "budget_heatmap"))
    args = ap.parse_args()

    cfg = DynaLoopConfig(enabled=True, keep_loops=1, window_size=64,
                         min_budget_ratio=0.02, block_size=1, chunk_size=5,
                         min_prompt_tokens=192, allocator="js",
                         js_kappa=args.js_kappa, js_gamma=args.js_gamma,
                         js_floor=args.js_floor)

    model, tok, _ = load_ouro(args.model, device=args.device)
    report = {"config": cfg.describe(), "model": args.model, "tasks": {}}
    for name, prompts in build_prompts(tok, args).items():
        mat, ret, kv, ctx = measure(model, tok, cfg, prompts, args.device)
        report["tasks"][name] = {"per_unit_retention": mat,
                                 "mean_retention": round(ret, 4),
                                 "kv_bytes_per_sample": kv,
                                 "ctx_tokens": ctx}
        print(f"{name:24s} retention={ret:.4f} ctx={ctx} "
              f"kv={kv / 1024 ** 2:.1f} MiB", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("saved ->", args.out + ".json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # noqa: BLE001
        print("matplotlib unavailable, skipping figure:", e)
        return

    names = list(report["tasks"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6),
                             constrained_layout=True)
    for ax, name in zip(axes, names):
        m = report["tasks"][name]["per_unit_retention"]
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"{name}\nmean retention "
                     f"{report['tasks'][name]['mean_retention']:.3f}",
                     fontsize=10)
        ax.set_xlabel("layer")
        if ax is axes[0]:
            ax.set_ylabel("loop")
    fig.colorbar(im, ax=axes, shrink=0.85, label="KV retention ratio")
    fig.savefig(args.out + ".png", dpi=170)
    print("saved ->", args.out + ".png")


if __name__ == "__main__":
    main()
