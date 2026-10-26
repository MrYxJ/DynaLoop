"""Capture Ouro's attention weights on a single math / QA sample.

Ouro runs ``total_ut_steps`` (4) *loops* over the same ``num_hidden_layers``
(24) stack, so one prefill executes 96 attention blocks in total.  This probe
records, for **every** (loop, layer) pair:

* ``p_last``  -- attention distribution of the **last** prompt token's query
  (averaged over heads).  This is the distribution that actually decides the
  first generated token, so it is the one DynaLoop must preserve.
* ``p_mass``  -- mean-received attention over all query positions (how much
  total attention mass each key position collects).
* ``ph_last`` -- same as ``p_last`` but per head.
* ``maps``    -- full down-sampled head x query x key attention map for the
  last layer (for heat-map figures).

Run::

    python scripts/attn_loop_probe.py --target-prompt 2000 --doc-idx 0

Output: ``<out-dir>/<task>_d<idx>_{meta.json,attn.npz}``
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

def _patch_mask_factories():
    """Make Ouro's modelling code work with transformers >= 5.

    Ouro ships ``input_embeds`` (a typo) in ``mask_kwargs``; transformers
    renamed it to ``inputs_embeds`` and started rejecting unknown kwargs
    outright.  ``modeling_ouro`` imports the factories at module load time, so
    the wrapper has to be installed *before* ``load_ouro`` pulls the model in.
    """
    import inspect
    try:
        from transformers import masking_utils as _mu
    except Exception:
        return
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        orig = getattr(_mu, name, None)
        if orig is None:
            continue

        def _wrap(orig=orig):
            def f(*a, **kw):
                try:
                    params = inspect.signature(orig).parameters
                except Exception:
                    params = None
                if params is not None:
                    # only rename when the installed version actually wants
                    # the corrected spelling (transformers 5.x) -- the pinned
                    # D:\Code\dynaloop_deps copy still wants `input_embeds`.
                    if ("input_embeds" in kw
                            and "input_embeds" not in params
                            and "inputs_embeds" in params):
                        kw["inputs_embeds"] = kw.pop("input_embeds")
                    if not any(p.kind == inspect.Parameter.VAR_KEYWORD
                               for p in params.values()):
                        kw = {k: v for k, v in kw.items() if k in params}
                return orig(*a, **kw)
            return f
        setattr(_mu, name, _wrap())


_patch_mask_factories()

from dynaloop import load_ouro  # noqa: E402


# --------------------------------------------------------------------------- #
# prompt                                                                      #
# --------------------------------------------------------------------------- #

def build_prompt(tok, task_name: str, n: int, seed: int, doc_idx: int,
                 target_tokens: int, needles: int = 16):
    """Rebuild exactly the prompt the harness would feed the model."""
    import run_math as L

    if task_name == "niah":
        # multi-needle NIAH, byte-for-byte the prompt the harness builds
        from dynaloop.eval.niah import build_niah_multi, niah_prompt
        it = build_niah_multi(target_tokens, needles, seed=5000 + doc_idx)
        text = niah_prompt(it)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        doc = {"question": it["question"], "target": it["answer"]}
        return text, ids, doc, 0

    t_gsm, t_math = L.build_tasks(n, seed, nshot=8,
                                  target_tokens=target_tokens, tok=tok,
                                  max_shots=64, verbose=True)
    task = t_math if task_name == "math500" else t_gsm
    docs = task.test_docs()
    doc = docs[doc_idx]
    k = int(task.config["num_fewshot"])
    ctx = task.fewshot_context(doc, k, rnd=random.Random(seed))
    text = ctx + L.GEN_PROMPT
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text, ids, doc, k


# --------------------------------------------------------------------------- #
# capture                                                                     #
# --------------------------------------------------------------------------- #

class Probe:
    """Wraps every ``OuroAttention.forward`` and records its attention map."""

    def __init__(self, model, ut_steps: int, n_layers: int,
                 map_layer: int, downsample: int = 4):
        self.model = model
        self.ut_steps = ut_steps
        self.n_layers = n_layers
        self.map_layer = map_layer
        self.ds = downsample
        self.L = None
        self._handles = []
        self._orig = []

    # ---- internals --------------------------------------------------- #
    def _record(self, ut, layer, w):
        # w: [1, H, Q, K] float32
        H, Q, K = w.shape[1], w.shape[2], w.shape[3]
        if self.L is None:
            self.L = K
            shape = (self.ut_steps, self.n_layers, K)
            self.p_last = np.zeros(shape, dtype=np.float32)
            self.p_mass = np.zeros(shape, dtype=np.float32)
            self.ph_last = np.zeros((self.ut_steps, self.n_layers, H, K),
                                    dtype=np.float32)
            self.entropy = np.zeros((self.ut_steps, self.n_layers),
                                    dtype=np.float32)
        A = w[0]                                  # [H, Q, K]
        self.p_last[ut, layer] = A[:, -1, :].mean(0).float().cpu().numpy()
        self.p_mass[ut, layer] = A.mean(0).mean(0).float().cpu().numpy()
        self.ph_last[ut, layer] = A[:, -1, :].float().cpu().numpy()
        p = A[:, -1, :].float().clamp_min(1e-12)
        self.entropy[ut, layer] = float(-(p * p.log()).sum(-1).mean().cpu())
        if layer == self.map_layer:
            m = F.avg_pool2d(A[:, None], (self.ds, self.ds))[:, 0]   # [H,Q',K']
            if self.maps is None:
                self.maps = np.zeros((self.ut_steps,) + tuple(m.shape),
                                     dtype=np.float16)
            self.maps[ut] = m.float().cpu().numpy().astype(np.float16)

    def __enter__(self):
        self.maps = None
        self.p_last = self.p_mass = self.ph_last = self.entropy = None

        def make(module, layer_idx):
            orig = module.forward

            def wrapped(*a, **kw):
                out = orig(*a, **kw)
                ut = int(kw.get("current_ut", 0))
                w = out[1] if isinstance(out, (tuple, list)) else None
                if w is not None:
                    self._record(ut, layer_idx, w.detach())
                return out

            return wrapped

        for li, dec in enumerate(self.model.model.layers):
            att = dec.self_attn
            self._orig.append(att.forward)
            att.forward = make(att, li)
        return self

    def __exit__(self, *exc):
        for li, dec in enumerate(self.model.model.layers):
            dec.self_attn.forward = self._orig[li]
        self._orig.clear()
        return False


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/Code/models/Ouro-1.4B")
    ap.add_argument("--task", default="math500",
                    choices=["math500", "gsm8k", "niah"])
    ap.add_argument("--needles", type=int, default=16,
                    help="niah only: number of hidden key->value needles")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--doc-idx", default="0",
                    help="comma separated doc indices, e.g. 0,1,2")
    ap.add_argument("--target-prompt", type=int, default=2000)
    ap.add_argument("--map-layer", type=int, default=23)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(ROOT, "results", "attn")
    os.makedirs(out_dir, exist_ok=True)

    model, tok, cfg = load_ouro(args.model, device=args.device)
    ut_steps = int(getattr(cfg, "total_ut_steps", 4))
    n_layers = int(cfg.num_hidden_layers)
    t_start = time.time()

    for doc_idx in [int(s) for s in args.doc_idx.split(",") if s.strip()]:
        text, ids, doc, kshot = build_prompt(tok, args.task, args.n, args.seed,
                                             doc_idx, args.target_prompt,
                                             needles=args.needles)
        print(f"[{args.task} d{doc_idx}] {len(ids)} tokens, "
              f"few-shot = {kshot}", flush=True)

        x = torch.tensor([ids], device=model.device)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
        with torch.no_grad(), Probe(model, ut_steps, n_layers,
                                    args.map_layer, args.downsample) as pr:
            model.model(input_ids=x, use_cache=False)
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                if args.device == "cuda" else 0.0)
        print(f"   forward {time.time() - t0:.1f}s, peak {peak:.2f} GiB",
              flush=True)

        tag = f"{args.task}_d{doc_idx}"
        npz = os.path.join(out_dir, f"{tag}_attn.npz")
        np.savez_compressed(
            npz, p_last=pr.p_last, p_mass=pr.p_mass,
            ph_last=pr.ph_last.astype(np.float16),
            entropy=pr.entropy, maps=pr.maps)
        meta = {
            "task": args.task, "doc_idx": doc_idx, "model": args.model,
            "n_tokens": len(ids), "fewshot": kshot,
            "target_prompt": args.target_prompt,
            "ut_steps": ut_steps, "n_layers": n_layers,
            "map_layer": args.map_layer, "downsample": args.downsample,
            "tokens": tok.convert_ids_to_tokens(ids),
            "problem": doc.get("question", ""),
            "gold": str(doc.get("target", "")),
            "npz": os.path.basename(npz),
            "text": text,
        }
        with open(os.path.join(out_dir, f"{tag}_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print("   saved ->", npz, flush=True)
    print(f"[all] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
