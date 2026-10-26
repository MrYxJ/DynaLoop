"""Batched multi-needle NIAH runner (same probes as ``run_eval.py``).

Identical task construction and grading, but all ``per_len`` prompts of a cell
are decoded together through :class:`batched_engine.BatchedOuroEngine`, which
amortises the weight-bandwidth cost that dominates looped-model decoding.

Usage
-----
    python scripts/run_niah_batched.py --rho 0.55 --chunk 5 --n 100 --batch 4 \
        --tag s4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dynaloop import (BatchedOuroEngine, DynaLoopConfig,  # noqa: E402
                       load_ouro)
from dynaloop.eval.niah import build_niah_multi, encode, niah_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DYNALOOP_MODEL",
                                                    "Ouro-1.4B"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="dynaloop", choices=["full", "dynaloop"])
    ap.add_argument("--rho", type=float, default=0.55)
    ap.add_argument("--keep-loops", type=int, default=2)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--bmin", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--allocator", default="block",
                    choices=["block", "anchor", "overlap", "fixed", "js"])
    ap.add_argument("--loop-ratios", default="",
                    help="fixed allocator: comma separated per-loop ratio, "
                         "e.g. '1,1,0.2,0.1'")
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="alpha: multiplier on the overlap budget")
    ap.add_argument("--js-gamma", type=float, default=1.0,
                    help="decay exponent of the rho->budget mapping")
    ap.add_argument("--js-kappa", type=float, default=1.0,
                    help="retention (= KV budget) of the first compressed loop")
    ap.add_argument("--js-floor", type=float, default=0.10,
                    help="retention floor for frozen loops/layers")
    ap.add_argument("--anchor-source", default="prev", choices=["prev", "loop1"])
    ap.add_argument("--min-prompt", type=int, default=192)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--needles", type=int, default=16)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    model, tok, _ = load_ouro(args.model, device=args.device)

    if args.mode == "full":
        cfg = DynaLoopConfig(enabled=False)
        name = "B0_fullkv"
    else:
        cfg = DynaLoopConfig(enabled=True, global_ratio=args.rho,
                             keep_loops=args.keep_loops,
                             window_size=args.window,
                             min_budget_ratio=args.bmin,
                             block_size=args.block,
                             chunk_size=args.chunk, loop_decay=args.decay,
                             min_prompt_tokens=args.min_prompt,
                             allocator=args.allocator,
                             theta=args.theta, top_frac=args.top_frac,
                             budget_scale=args.scale,
                             loop_ratios=args.loop_ratios,
                             js_gamma=args.js_gamma,
                             js_kappa=args.js_kappa, js_floor=args.js_floor,
                             anchor_source=args.anchor_source)
        name = (f"{('Anchor' if args.allocator == 'anchor' else 'DynaLoop')}"
                f"_rho{args.rho}_K{args.keep_loops}_ws{args.window}"
                f"_bmin{args.bmin}_m{args.block}_c{args.chunk}_d{args.decay}"
                + (f"_th{args.theta}_{args.anchor_source}"
                   if args.allocator == "anchor" else "")
                + (f"_ovl{args.top_frac}_a{args.scale}"
                   if args.allocator == "overlap" else "")
                + (f"_lr{args.loop_ratios.replace(',', '-')}"
                   if args.allocator == "fixed" else "")
                + (f"_js_g{args.js_gamma}"
                   f"_k{args.js_kappa}_f{args.js_floor}"
                   if args.allocator == "js" else ""))

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    nl_ids = set()
    for s in ("\n", "\n\n"):
        try:
            nl_ids.update(tok.encode(s, add_special_tokens=False))
        except Exception:
            pass

    items = [build_niah_multi(args.length, args.needles, seed=5000 + i)
             for i in range(args.n)]
    prompts = [niah_prompt(it) for it in items]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    hits = 0
    rets, kvs = [], []
    t_all = time.time()
    for start in range(0, len(order), args.batch):
        idxs = order[start:start + args.batch]
        enc = [encode(tok, prompts[i])[0] for i in idxs]
        lens = torch.tensor([len(x) for x in enc], device=args.device)
        S = int(lens.max().item())
        batch = torch.full((len(enc), S), pad_id, dtype=torch.long,
                           device=args.device)
        for j, x in enumerate(enc):
            batch[j, :len(x)] = x

        eng = BatchedOuroEngine(model, tok, cfg, device=args.device)
        seqs, valid = eng.generate(batch, lens,
                                   max_new_tokens=args.max_new,
                                   stop_ids=sorted(nl_ids))
        rets.extend(eng.per_sample_retention(lens))
        for j, i in enumerate(idxs):
            txt = tok.decode(seqs[j][:valid[j]], skip_special_tokens=True)
            gold = items[i]["answer"]
            ok = gold in txt.replace(",", "")
            hits += int(ok)
            log(f"  niah-multi len={args.length} k={args.needles} s={i} "
                f"hit={ok} ctx={int(lens[j])} | '{txt.strip()[:24]}'")
        kv = eng.kv_bytes() / max(len(idxs), 1)
        kvs.append(eng.kv_bytes())
        # per-loop retention (which loops actually shrink under this policy)
        if not hasattr(main, "_loop_fill"):
            main._loop_fill, main._loop_prompt = {}, {}
        for u, t in eng._prompt_fill.items():
            ut, l = u
            main._loop_fill[ut] = main._loop_fill.get(ut, 0.0) + float(t.sum())
        for j, i in enumerate(idxs):
            for ut in range(eng.H):
                # x eng.M: ``_loop_fill`` already sums the loop's M layers
                main._loop_prompt[ut] = (main._loop_prompt.get(ut, 0.0)
                                         + int(lens[j]) * eng.M)
        del eng
        torch.cuda.empty_cache()

    report = {
        "run": name + (("_" + args.tag) if args.tag else ""),
        "mode": args.mode, "config": cfg.describe(),
        "model": args.model, "time": time.ctime(),
        "niah_multi": {
            f"niahm_{args.length}_k{args.needles}": {
                "acc": round(hits / max(args.n, 1), 4),
                "n": args.n,
                "retention": round(sum(rets) / max(len(rets), 1), 4),
                "kv_MiB": round(sum(kvs) / max(len(kvs), 1) / 1024 ** 2, 2),
            }
        },
        "wall_s": round(time.time() - t_all, 1),
    }
    if getattr(main, "_loop_fill", None):
        report["per_loop_retention"] = {
            f"loop{ut + 1}": round(main._loop_fill.get(ut, 0.0)
                                  / max(main._loop_prompt.get(ut, 1.0), 1.0), 4)
            for ut in sorted(main._loop_fill)}
    out = os.path.join(ROOT, "results", f"{report['run']}_niahm.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(json.dumps(report, ensure_ascii=False, indent=2))
    log("saved -> " + out)


if __name__ == "__main__":
    main()
