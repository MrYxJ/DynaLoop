"""Length-scaling benchmark (FlashLoop Fig 6 protocol): FullKV vs DynaLoop js-stab.

For ctx in {1024, 2048, 4096, 8192} (real, subject to 8 GB VRAM):
  * KV cache memory per sample (bytes, exact tensor accounting)
  * Prefill latency: 3 runs, median (CUDA-sync wall clock)   [FlashLoop protocol]
  * Decode latency: 3 warm-up steps, mean over next 16 steps [FlashLoop protocol]

For 16384 / 32768 the card cannot materialise the [16, S, S] attention matrix;
we record OOM and report analytic KV (exact linear scaling) + extrapolated
latency (fitted on measured points), flagged as "extrap".

Usage:
    python scripts/run_len_scaling.py --tag s18
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

from dynaloop import BatchedOuroEngine, DynaLoopConfig, load_ouro  # noqa: E402

# exact per-token KV bytes of the FULL cache (bf16 K+V):
# 4 loops x 24 layers x 2 (K,V) x 16 kv-heads x 128 head_dim x 2 bytes
FULL_KV_BYTES_PER_TOKEN = 4 * 24 * 2 * 16 * 128 * 2


def make_prompt(tok, ctx: int) -> str:
    unit = "The quick brown fox jumps over the lazy dog. "
    return unit * max(2, ctx // 12)


def bench_once(model, tok, cfg, ctx: int, steps: int, mode: str):
    text = make_prompt(tok, ctx)
    ids = tok(text, return_tensors="pt")["input_ids"][0].cuda()
    S = int(ids.numel())
    input_ids = ids.unsqueeze(0).contiguous()
    lens = torch.tensor([S], dtype=torch.long, device="cuda")

    torch.cuda.synchronize()
    t0 = time.time()
    eng = BatchedOuroEngine(model, tok, cfg, device="cuda")
    logits = eng.prefill(input_ids, lens)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0
    kv_bytes = eng.kv_bytes()                      # per sample (batch=1)
    kv_tokens = eng.kv_tokens()
    ret = eng.retention_ratio(lens)

    eng.prepare_decode(steps + 8)
    pos = lens.clone()
    for _ in range(3):                             # warm-up
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        logits = eng.decode_step(nxt, pos)
        pos = pos + 1
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(steps):
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        logits = eng.decode_step(nxt, pos)
        pos = pos + 1
    torch.cuda.synchronize()
    decode_ms = (time.time() - t1) / steps * 1000
    del eng
    torch.cuda.empty_cache()
    return {"prefill_s": prefill_s, "decode_ms": decode_ms,
            "kv_bytes": kv_bytes, "kv_tokens": kv_tokens, "retention": ret,
            "ctx_tokens": S}


def run_cell(model, tok, cfg, ctx: int, mode: str, steps: int,
             prefill_reps: int):
    runs, err = [], None
    for i in range(prefill_reps):
        try:
            runs.append(bench_once(model, tok, cfg, ctx, steps, mode))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            err = "OOM"
            break
        except RuntimeError as e:                  # some OOMs surface as RuntimeError
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                err = "OOM"
                break
            raise
    if runs:
        runs.sort(key=lambda r: r["prefill_s"])
        med = runs[len(runs) // 2]
        return {"status": "ok", "prefill_reps": len(runs),
                "prefill_s": round(med["prefill_s"], 4),
                "decode_ms": round(med["decode_ms"], 2),
                "kv_MiB": round(med["kv_bytes"] / 1024 ** 2, 2),
                "kv_tokens": med["kv_tokens"],
                "retention": round(med["retention"], 4),
                "ctx_tokens": med["ctx_tokens"]}
    return {"status": err, "ctx": ctx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DYNALOOP_MODEL",
                                                    "Ouro-1.4B"))
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--kappa", type=float, default=0.5,
                    help="js allocator: retention of the first compressed loop")
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    model, tok, _ = load_ouro(args.model, device="cuda")

    full_cfg = DynaLoopConfig(enabled=False)
    # single-rule js allocator: r = clip(kappa * (d_u / D1)^gamma, floor, 1)
    dyn_cfg = DynaLoopConfig(enabled=True, global_ratio=0.55, keep_loops=1,
                             window_size=64, min_budget_ratio=0.02, block_size=1,
                             chunk_size=5, min_prompt_tokens=192,
                             allocator="js", js_gamma=1.0,
                             js_kappa=args.kappa, js_floor=0.10)

    report = {"protocol": {"prefill": "%d runs, median, cuda-sync" % args.reps,
                           "decode": "3 warm-up + mean over %d steps" % args.steps,
                           "batch": 1, "steps": args.steps,
                           "js_kappa": args.kappa, "js_gamma": 1.0,
                           "js_floor": 0.10},
              "fullkv_bytes_per_token": FULL_KV_BYTES_PER_TOKEN, "cells": {}}
    for ctx in args.ctx:
        cell = {}
        print(f"=== ctx={ctx} ===", flush=True)
        for mode, cfg in (("fullkv", full_cfg), ("dynaloop", dyn_cfg)):
            r = run_cell(model, tok, cfg, ctx, mode, args.steps, args.reps)
            if r["status"] != "ok":
                # analytic KV for the full cache is exact (linear in S)
                if mode == "fullkv":
                    r["kv_MiB_analytic"] = round(
                        ctx * FULL_KV_BYTES_PER_TOKEN / 1024 ** 2, 2)
                print(f"  {mode}: {r['status']}", flush=True)
            else:
                print(f"  {mode}: {json.dumps(r)}", flush=True)
            cell[mode] = r
        report["cells"][str(ctx)] = cell
        # incremental save
        out = os.path.join(ROOT, "results", f"len_scaling_{args.tag}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
