"""Minimal end-to-end example: FullKV vs DynaLoop on one prompt.

    python examples/quickstart.py

Prints, for each configuration, the generated answer together with the
realised prompt-KV footprint (bytes, tokens, retention) so you can see what
the policy actually does on a single short run (~1 minute on an 8 GB GPU).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dynaloop import BatchedOuroEngine, DynaLoopConfig, load_ouro  # noqa: E402
from dynaloop.eval.datasets import GSM8K_FEWSHOT, build_chat_prompt  # noqa: E402

QUESTION = ("Betty is saving money for a new wallet which costs $100. Betty "
            "has only half of the money she needs. Her parents decided to "
            "give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need?")

CONFIGS = {
    # reference: nothing is evicted
    "FullKV": dict(enabled=False),
    # our rule: 1 loop full, loops 2-4 get kappa * (JS / D1)^gamma
    "DynaLoop (kappa=0.5, gamma=1)": dict(
        enabled=True, keep_loops=1, window_size=64, min_budget_ratio=0.02,
        block_size=1, chunk_size=5, min_prompt_tokens=192, allocator="js",
        js_kappa=0.5, js_gamma=1.0, js_floor=0.10),
    # more aggressive on the first compressed loop
    "DynaLoop (kappa=0.25, gamma=1)": dict(
        enabled=True, keep_loops=1, window_size=64, min_budget_ratio=0.02,
        block_size=1, chunk_size=5, min_prompt_tokens=192, allocator="js",
        js_kappa=0.25, js_gamma=1.0, js_floor=0.10),
}


def run(model, tok, cfg_kwargs, prompt, device, max_new_tokens=160):
    cfg = DynaLoopConfig(**cfg_kwargs)
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    lens = torch.tensor([ids.shape[1]], device=device)
    eng = BatchedOuroEngine(model, tok, cfg, device=device)
    seqs, valid = eng.generate(ids, lens, max_new_tokens=max_new_tokens,
                               stop_ids=[tok.eos_token_id])
    kv = int(eng.kv_bytes())
    out = tok.decode(seqs[0][:valid[0]], skip_special_tokens=True)
    res = {
        "text": out.strip(),
        "ctx_tokens": int(ids.shape[1]),
        "prompt_kv_MiB": round(kv / 1024 ** 2, 2),
        "retention": round(float(eng.retention_ratio(lens)), 4),
        "per_loop_retention": eng.per_loop_retention(lens),
    }
    del eng
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DYNALOOP_MODEL",
                                                      "Ouro-1.4B"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()

    model, tok, _ = load_ouro(args.model, device=args.device)
    prompt = build_chat_prompt(tok, QUESTION, task="gsm8k",
                               fewshot=GSM8K_FEWSHOT)
    print(f"question : {QUESTION}\ncontext  : "
          f"{len(tok(prompt)['input_ids'])} tokens\n", flush=True)

    for name, kwargs in CONFIGS.items():
        r = run(model, tok, kwargs, prompt, args.device, args.max_new)
        print(f"--- {name}")
        print(f"    KV (prompt) : {r['prompt_kv_MiB']:.2f} MiB   "
              f"retention {r['retention']:.3f}")
        print(f"    per-loop    : {r['per_loop_retention']}")
        print(f"    answer      : {r['text'][:200]}\n", flush=True)


if __name__ == "__main__":
    main()
