"""FlashLoop-style evaluation of Ouro-1.4B through the EleutherAI LM Harness.

FlashLoop (§5.1) evaluates MATH-500 / GSM8K / ARC-C / HellaSwag / WinoGrande
"using the EleutherAI LM Harness".  This module reproduces that protocol for
our own KV-cache policies: it registers our Ouro + DynaLoop engine as an
``lm_eval`` model (``ouro-dynaloop``) and two *local* tasks (``gsm8k_local``,
``math500_local``) that read the already-downloaded datasets from disk, so the
run needs no network.

Why local tasks instead of the upstream ``gsm8k`` / ``math_500`` configs
---------------------------------------------------------------------
The upstream configs pull their data from the HF hub, which is unreachable
here.  We therefore implement the old-style ``lm_eval.api.task.Task``
interface directly, keeping everything the harness provides (``num_fewshot``
exemplar sampling, ``--samples`` index selection, request batching, metric
aggregation) while reading the exact same files we already use elsewhere.

Prompt format
-------------
Ouro is an *instruct* model: a plain "Question: ... Answer:" prompt makes it
continue the question instead of answering.  The harness concatenates
``doc_to_text`` + ``doc_to_target`` pairs with ``\\n\\n``, so we emit ChatML
turns from those two hooks -- which turns the standard few-shot concatenation
into a proper multi-turn chat prompt (the harness's own
``apply_chat_template``/``fewshot_as_multiturn`` path is only available to the
newer ``ConfigurableTask`` YAML configs).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import lm_eval  # noqa: E402
from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.api.metrics import mean  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.api.task import Task  # noqa: E402

from dynaloop import BatchedOuroEngine, DynaLoopConfig, load_ouro  # noqa: E402
from dynaloop.eval.datasets import (ARC_C_TRAIN_PATH, GSM8K_FEWSHOT_8,
                                    HELLASWAG_TRAIN_PATH, MATH_FEWSHOT_8,
                                    clean_gsm_solution, eval_indices,
                                    extract_choice, extract_gsm8k,
                                    extract_math, fewshot_pools, is_equiv,
                                    load_arc_challenge, load_gsm8k,
                                    load_hellaswag, load_math500,
                                    normalize_answer)

SYS_TURN = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
GEN_PROMPT = "<|im_start|>assistant\n"
SUFFIX_GSM = ("\nWork through the reasoning step by step, then give the final "
              "numeric answer on the last line as: Answer: <number>")
SUFFIX_MATH = ("\nWork through the reasoning step by step, then give the "
               "final answer inside \\boxed{}.")

_LETTERS = "ABCDEFGH"


def _mc_options(options) -> str:
    """Render ``["...", ...]`` as ``A. ...\\nB. ...``."""
    return "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(options))


def _user(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n"


def _asst(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #

@register_model("ouro-dynaloop")
class OuroDynaLoopLM(LM):
    """Ouro-1.4B driven by our own looped-KV engine, exposed to lm-eval."""

    def __init__(self, pretrained="D:/Code/models/Ouro-1.4B", mode="full",
                 rho=0.55, keep_loops=2, chunk=5, window=64, bmin=0.05,
                 block=6, min_prompt=192, batch_size=4, max_gen_toks=512,
                 allocator="block", theta=0.9, top_frac=0.10, budget_scale=1.0,
                 anchor_source="prev", loop_ratios="",
                 js_gamma=1.0, js_kappa=1.0, js_floor=0.10,
                 device="cuda", gen_prompt="<|im_start|>assistant\n", **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.gen_prompt = gen_prompt
        self._device = device
        self.batch_size = int(batch_size)
        self.max_gen_toks = int(max_gen_toks)
        self.max_length = 4096
        self.model, self.tok, _ = load_ouro(pretrained, device=device)
        self.pad_id = self.tok.pad_token_id or 0
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eot_token_id = im_end if isinstance(im_end, int) and im_end >= 0 else 0
        if mode == "full":
            self.cfg = DynaLoopConfig(enabled=False)
        else:
            keep = 0 if mode == "dynamickv" else int(keep_loops)
            self.cfg = DynaLoopConfig(enabled=True, global_ratio=float(rho),
                                      keep_loops=keep, window_size=int(window),
                                      min_budget_ratio=float(bmin),
                                      block_size=int(block),
                                      chunk_size=int(chunk),
                                      min_prompt_tokens=int(min_prompt),
                                      allocator=str(allocator),
                                      theta=float(theta),
                                      top_frac=float(top_frac),
                                      budget_scale=float(budget_scale),
                                      loop_ratios=str(loop_ratios),
                                      js_gamma=float(js_gamma),
                                      js_kappa=float(js_kappa),
                                      js_floor=float(js_floor),
                                      anchor_source=str(anchor_source))
        self.mode = mode
        # efficiency bookkeeping (not part of the harness API)
        self.stats = {"kv_bytes": 0.0, "n": 0, "prefill_s": 0.0,
                      "decode_s": 0.0, "decode_steps": 0}

    # -- LM API ---------------------------------------------------------- #
    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self):
        return "Ouro-1.4B"

    def tok_encode(self, string: str, **kwargs):
        # The context already contains the ChatML turns; ``gen_prompt`` opens
        # the assistant turn, which the harness's plain-text few-shot
        # concatenation cannot add for us.
        return self.tok(string + self.gen_prompt,
                        add_special_tokens=False)["input_ids"]

    def loglikelihood(self, requests):
        raise NotImplementedError("generate_until only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("generate_until only")

    def generate_until(self, requests):
        enc = [self.tok_encode(r.arguments[0]) for r in requests]
        order = sorted(range(len(requests)), key=lambda i: len(enc[i]))
        out = [None] * len(requests)
        for start in range(0, len(order), self.batch_size):
            idxs = order[start:start + self.batch_size]
            try:
                self._run_group(idxs, enc, requests, out)
            except torch.OutOfMemoryError:
                # A 2000-token FullKV batch is right at the 8 GB ceiling; when
                # it does not fit, fall back to one sample at a time instead of
                # losing the whole run.
                del idxs
                torch.cuda.empty_cache()
                msg = ("[ouro-dynaloop] OOM on a batch of "
                       f"{min(self.batch_size, len(order) - start)}; "
                       "retrying sample-by-sample")
                print(msg, flush=True)
                self.stats["oom_fallbacks"] = self.stats.get("oom_fallbacks", 0) + 1
                for i in order[start:start + self.batch_size]:
                    self._run_group([i], enc, requests, out)
        return out

    def _run_group(self, idxs, enc, requests, out):
        seqs = [enc[i] for i in idxs]
        S = max(len(x) for x in seqs)
        pad = torch.full((len(seqs), S), self.pad_id, dtype=torch.long,
                         device=self._device)
        lens = torch.tensor([len(x) for x in seqs], device=self._device)
        for j, x in enumerate(seqs):
            pad[j, :len(x)] = torch.tensor(x, device=self._device)
        mx = min(max(int(r.arguments[1].get("max_gen_toks",
                                             self.max_gen_toks))
                     for r in (requests[i] for i in idxs)),
                 self.max_gen_toks)
        eng = BatchedOuroEngine(self.model, self.tok, self.cfg,
                                device=self._device)
        seqs_o, valid = eng.generate(pad, lens, max_new_tokens=mx,
                                     stop_ids=[self.eot_token_id])
        self.stats["kv_bytes"] += eng.kv_bytes()
        self.stats["n"] += len(idxs)
        self.stats["prefill_s"] += eng.stats.get("prefill_cost", 0.0)
        self.stats["decode_s"] += eng.stats.get("decode_cost", 0.0)
        self.stats["decode_steps"] += int(eng.stats.get("generated", 0))
        pr = getattr(eng, "_prompt_loop_ret", None)
        if pr:
            self.stats.setdefault("loop_ret", []).append(pr)
        for j, i in enumerate(idxs):
            txt = self.tok.decode(seqs_o[j][:valid[j]],
                                  skip_special_tokens=True)
            for u in (requests[i].arguments[1].get("until") or []):
                if u and u in txt:
                    txt = txt.split(u)[0]
            out[i] = txt
        del eng
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# tasks                                                                       #
# --------------------------------------------------------------------------- #

class _LocalTask(Task):
    VERSION = 0
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config=None, docs=None, fewshot=None, suffix="",
                 gen_toks=0, **kw):
        self._docs = list(docs or [])
        self._fewshot = list(fewshot or [])
        self._suffix = suffix
        self.gen_toks = int(gen_toks)
        super().__init__(config=config)

    # data comes from disk, not from the hub
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        return None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self._fewshot

    def test_docs(self):
        return self._docs

    def doc_to_target(self, doc):
        return _asst(doc["target"])

    def doc_to_prefix(self, doc):
        """Text appended right before generation (unused: we open the
        assistant turn inside the LM wrapper)."""
        return ""

    def construct_requests(self, doc, ctx, metadata=None, max_gen_toks=512,
                           **kwargs):
        # ``kwargs`` carries apply_chat_template / chat_template / gen_prefix;
        # we build the ChatML prompt ourselves, so they are intentionally
        # dropped (and must NOT be forwarded to Instance).  ``metadata`` is
        # (task_name, doc_id, repeats) and *must* be forwarded -- the evaluator
        # groups instances by doc_id and multiplies by repeats.
        if metadata is None:
            metadata = (self.config["task"], None, 1)
        metadata = (metadata[0], metadata[1], metadata[2] or 1)
        mx = int(self.gen_toks or max_gen_toks)
        return Instance(request_type="generate_until", doc=doc,
                        arguments=(ctx, {"until": ["<|im_end|>", "Question:"],
                                         "do_sample": False,
                                         "max_gen_toks": mx}),
                        idx=0, metadata=metadata)

    def process_results(self, doc, results):
        raise NotImplementedError

    def aggregation(self):
        return {"exact_match": mean}

    def higher_is_better(self):
        return {"exact_match": True}


def _first_text(results) -> str:
    """lm-eval hands us ``[filtered_resps[filter_key]]`` -> a list of lists."""
    if not results:
        return ""
    r = results[0]
    if isinstance(r, (list, tuple)):
        r = r[0] if r else ""
    return r if isinstance(r, str) else str(r)


class GSM8KLocal(_LocalTask):
    def doc_to_text(self, doc):
        return _user(doc["question"] + self._suffix)

    def process_results(self, doc, results):
        pred = extract_gsm8k(_first_text(results))
        gold = doc["target"]
        ok = pred is not None and (normalize_answer(pred) == normalize_answer(gold)
                                   or is_equiv(pred, gold))
        return {"exact_match": 1.0 if ok else 0.0}


class Math500Local(_LocalTask):
    def doc_to_text(self, doc):
        return _user(doc["question"] + self._suffix)

    def process_results(self, doc, results):
        pred = extract_math(_first_text(results))
        gold = doc["target"]
        ok = pred is not None and (normalize_answer(pred) == normalize_answer(gold)
                                   or is_equiv(pred, gold))
        return {"exact_match": 1.0 if ok else 0.0}


class ArccLocal(_LocalTask):
    """ARC-Challenge (1172 test items), graded on the generated letter.

    The upstream lm-eval config also reports ``acc_norm`` from choice
    log-likelihoods, but our engine has no log-likelihood path, so both the
    baseline and the compressed run are graded identically from what the model
    writes -- which is exactly the comparison we care about.
    """

    def doc_to_text(self, doc):
        q = f"Question: {doc['question']}\n" + _mc_options(doc["choices"])
        return _user(q + "\nAnswer with just the letter of the correct option.")

    def process_results(self, doc, results):
        pred = extract_choice(_first_text(results), n_opts=len(doc["choices"]))
        ok = pred is not None and pred == doc["target"]
        return {"exact_match": 1.0 if ok else 0.0}


class HellaSwagLocal(_LocalTask):
    """HellaSwag (10042 validation items), graded on the generated letter."""

    def doc_to_text(self, doc):
        body = f"{doc['ctx']}\n" + _mc_options(doc["endings"])
        return _user("Which continuation is the most plausible?\n\n" + body +
                     "\nAnswer with just the letter of the correct option.")

    def process_results(self, doc, results):
        pred = extract_choice(_first_text(results), n_opts=len(doc["endings"]))
        ok = pred is not None and pred == doc["target"]
        return {"exact_match": 1.0 if ok else 0.0}


# --------------------------------------------------------------------------- #
# driver                                                                      #
# --------------------------------------------------------------------------- #

def build_tasks(n: int, seed: int, nshot: int, target_tokens: int = 0,
                tok=None, max_shots: int = 64, verbose: bool = True,
                offset: int = 0, total: int = 0,
                which=("gsm8k", "math500")):
    """Build the local math tasks (only the ones named in ``which``).

    ``target_tokens > 0`` switches on *automatic* shot selection: exemplars are
    drawn from a held-out pool of real worked solutions until the rendered
    ChatML context reaches ``target_tokens``.  KV-cache compression only has
    something to bite on once the prompt is long, so instead of hard-coding a
    shot count we grow it until the measured length hits the target.  This keeps
    every method at exactly the same prompt length, which is what a fair
    FullKV / DynamicKV / DynaLoop comparison requires.
    """
    gsm, math = load_gsm8k(), load_math500()
    # Chunked evaluation support: draw the full index list once, then slice
    # [offset : offset+n].  The few-shot pool below still excludes ALL n_total
    # eval items, so a chunk sees exactly the same prompts as a full run.
    # NOTE: eval_indices() draws a sample of size k, so subsets of different
    # sizes are NOT nested.  Always draw `total` (the full eval set) and slice,
    # otherwise chunk 0 of a chunked run would see different problems.
    n_total = total or n
    gi = eval_indices(len(gsm), n_total, seed)[offset:offset + n]
    mi = eval_indices(len(math), n_total, seed)[offset:offset + n]
    gsm_docs = [{"question": gsm[i]["question"], "target": gsm[i]["answer"]}
                for i in gi]
    math_docs = [{"question": math[i]["problem"], "target": math[i]["answer"]}
                 for i in mi]
    gsm_pool, math_pool = fewshot_pools(exclude_gsm=gi, exclude_math=mi)

    gsm_fs = [{"question": q, "target": a} for q, a in GSM8K_FEWSHOT_8[:nshot]]
    math_fs = [{"question": q, "target": a} for q, a in MATH_FEWSHOT_8[:nshot]]

    specs = []
    if "gsm8k" in which:
        t_gsm = GSM8KLocal(config={"task": "gsm8k_local", "num_fewshot": nshot,
                                   "output_type": "generate_until"},
                           docs=gsm_docs, fewshot=gsm_fs, suffix=SUFFIX_GSM)
        t_gsm._task_name = "gsm8k_local"
        specs.append((t_gsm, gsm_pool))
    if "math500" in which:
        t_math = Math500Local(config={"task": "math500_local",
                                      "num_fewshot": nshot,
                                      "output_type": "generate_until"},
                              docs=math_docs, fewshot=math_fs,
                              suffix=SUFFIX_MATH)
        t_math._task_name = "math500_local"
        specs.append((t_math, math_pool))
    tasks = tuple(t for t, _ in specs)
    # How many exemplars each task actually ended up with.  We track it here
    # rather than reading it back from the task, because lm_eval nulls
    # ``num_fewshot`` out when a config dict is passed at construction time.
    shots = {id(t): nshot for t in tasks}

    if target_tokens and tok is not None:
        for t, pool in specs:
            k, ctxlen = _grow_fewshot(t, pool, tok, seed, target_tokens,
                                      max_shots)
            shots[id(t)] = k
            if verbose:
                print(f"[{t.__class__.__name__}] few-shot = {k} exemplars, "
                      f"context = {ctxlen} tokens (target {target_tokens})",
                      flush=True)
    for t in tasks:
        # reset so ``evaluate`` re-samples the very same exemplars we measured
        t.set_config("num_fewshot", int(shots[id(t)]))
        t.fewshot_rnd = random.Random(seed)
    return tasks


ALL_TASKS = ("gsm8k", "math500", "arcc", "hellaswag")


def build_mc_tasks(n: int, seed: int, tok=None, target_tokens: int = 0,
                   max_shots: int = 64, verbose: bool = True,
                   offset: int = 0, total: int = 0, which=("arcc", "hellaswag")):
    """Build the multiple-choice tasks (ARC-Challenge / HellaSwag).

    Same contract as :func:`build_tasks`: a fixed random subset, few-shot
    exemplars drawn from the datasets' own train splits, and shot count grown
    until the rendered context reaches ``target_tokens`` when requested.
    """
    n_total = total or n
    out = []
    specs = []
    if "arcc" in which:
        arc = load_arc_challenge()
        arc_train = load_arc_challenge(ARC_C_TRAIN_PATH)
        ai = eval_indices(len(arc), n_total, seed)[offset:offset + n]
        docs = [{"question": arc[i]["question"],
                 "choices": arc[i]["choices"],
                 "target": arc[i]["answer"]} for i in ai]
        pool = [{"question": r["question"], "choices": r["choices"],
                 "target": r["answer"]} for r in arc_train]
        specs.append(("arcc_local", ArccLocal, docs, pool))
    if "hellaswag" in which:
        hs = load_hellaswag()
        hs_train = load_hellaswag(HELLASWAG_TRAIN_PATH)
        hi = eval_indices(len(hs), n_total, seed)[offset:offset + n]
        docs = [{"ctx": hs[i]["ctx"], "endings": hs[i]["endings"],
                 "target": hs[i]["answer"]} for i in hi]
        pool = [{"ctx": r["ctx"], "endings": r["endings"],
                 "target": r["answer"]} for r in hs_train]
        specs.append(("hellaswag_local", HellaSwagLocal, docs, pool))

    tasks = []
    for name, cls, docs, pool in specs:
        # num_fewshot starts at 1 (0 is treated as None by Task.set_config);
        # _grow_fewshot rewrites it with the real count below.
        t = cls(config={"task": name, "num_fewshot": 1,
                        "output_type": "generate_until"},
                docs=docs, fewshot=[], suffix="")
        tasks.append(t)
        t._pool = pool
        # lm_eval stores the raw config dict under ``config["task"]``, so the
        # readable name has to ride along separately.
        t._task_name = name
    # Exemplar count per task.  Tracked here rather than read back from the
    # task, because lm_eval nulls ``num_fewshot`` when a config dict is
    # passed at construction time.
    shots = {id(t): 1 for t in tasks}
    if target_tokens and tok is not None:
        for t in tasks:
            k, ctxlen = _grow_fewshot(t, t._pool, tok, seed, target_tokens,
                                      max_shots)
            shots[id(t)] = k
            if verbose:
                print(f"[{t.__class__.__name__}] few-shot = {k} exemplars, "
                      f"context = {ctxlen} tokens (target {target_tokens})",
                      flush=True)
    for t in tasks:
        k = int(shots[id(t)])
        t.set_config("num_fewshot", k)
        # ``Task.fewshot_examples`` draws from ``_training_docs``, so the pool
        # has to be handed over explicitly: we construct the task with
        # ``fewshot=[]``, which leaves it empty.  _grow_fewshot already did
        # this when it ran.
        if k > 0 and not getattr(t, "_training_docs", None):
            t._fewshot = list(t._pool[:k])
            t._training_docs = list(t._pool[:k])
        t.fewshot_rnd = random.Random(seed)
        out.append(t)
    return tuple(out)


def _grow_fewshot(task, pool, tok, seed, target_tokens, max_shots):
    """Add exemplars one at a time until the rendered context hits target."""
    probe = task.test_docs()[0]
    last, lastlen = 1, 0
    for k in range(1, min(max_shots, len(pool)) + 1):
        # ``fewshot_examples`` reads the cached ``_training_docs`` (populated in
        # Task.__init__), so re-assigning ``_fewshot`` alone would be ignored.
        task._fewshot = pool[:k]
        task._training_docs = pool[:k]
        task.set_config("num_fewshot", k)
        ctx = task.fewshot_context(probe, k, rnd=random.Random(seed))
        ntok = len(tok(ctx + GEN_PROMPT, add_special_tokens=False)["input_ids"])
        last, lastlen = k, ntok
        if ntok >= target_tokens:
            return k, ntok
    return last, lastlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DYNALOOP_MODEL",
                                                    "Ouro-1.4B"))
    ap.add_argument("--mode", default="full",
                    choices=["full", "dynamickv", "dynaloop"])
    ap.add_argument("--rho", type=float, default=0.55)
    ap.add_argument("--keep-loops", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--bmin", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--allocator", default="block",
                    choices=["block", "anchor", "overlap", "fixed", "js"],
                    help="block=DynamicKV Phi; anchor=coverage; overlap=top-k "
                         "set overlap; fixed=hand-written ratios; js=JS div")
    ap.add_argument("--loop-ratios", default="",
                    help="fixed allocator: comma separated per-loop ratio")
    ap.add_argument("--js-gamma", type=float, default=1.0,
                    help="decay exponent: how fast later loops fall from "
                         "kappa to the floor (1 = fall straight down, "
                         "< 1 = later loops stay wider)")
    ap.add_argument("--js-kappa", type=float, default=1.0,
                    help="retention of the first compressed loop (the one "
                         "with rho ~ 1): this IS its KV budget")
    ap.add_argument("--js-floor", type=float, default=0.10,
                    help="retention floor for loops/layers whose attention "
                         "has frozen")
    ap.add_argument("--theta", type=float, default=0.9,
                    help="coverage target of the anchor allocator")
    ap.add_argument("--top-frac", type=float, default=0.10,
                    help="size of the important set for the overlap allocator")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="alpha: multiplier on the overlap budget")
    ap.add_argument("--anchor-source", default="prev", choices=["prev", "loop1"],
                    help="anchor allocator reference: prev = previous loop, "
                         "loop1 = always loop 1")
    ap.add_argument("--min-prompt", type=int, default=192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-gen-toks", type=int, default=512)
    ap.add_argument("--gen-toks", type=int, default=0,
                    help="per-task max_gen_toks actually requested from the "
                         "model (0 = harness default, 256)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many eval items (chunked runs)")
    ap.add_argument("--total", type=int, default=0,
                    help="full eval-set size to draw from before slicing "
                         "with --offset (keeps chunks identical to one run)")
    ap.add_argument("--nshot", type=int, default=8,
                    help="ignored when --target-prompt > 0")
    ap.add_argument("--target-prompt", type=int, default=0,
                    help="grow the few-shot exemplars until the rendered "
                         "context reaches this many tokens (e.g. 2000)")
    ap.add_argument("--max-shots", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="both",
                    choices=["both", "gsm8k", "math500", "mc", "arcc",
                             "hellaswag"],
                    help="mc = ARC-C + HellaSwag (generation-based, because "
                         "the engine has no log-likelihood path)")
    ap.add_argument("--tag", default="lmeval")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    lm = OuroDynaLoopLM(pretrained=args.model, mode=args.mode, rho=args.rho,
                        keep_loops=args.keep_loops, chunk=args.chunk,
                        window=args.window, bmin=args.bmin, block=args.block,
                        min_prompt=args.min_prompt, batch_size=args.batch,
                        max_gen_toks=args.max_gen_toks,
                        allocator=args.allocator, theta=args.theta,
                        top_frac=args.top_frac, budget_scale=args.scale,
                        loop_ratios=args.loop_ratios,
                        js_gamma=args.js_gamma,
                        js_kappa=float(getattr(args, "js_kappa", 1.0)),
                        js_floor=float(getattr(args, "js_floor", 0.10)),
                        anchor_source=args.anchor_source)
    if args.task in ("mc", "arcc", "hellaswag"):
        which = ("arcc", "hellaswag") if args.task == "mc" else (args.task,)
        all_tasks = list(build_mc_tasks(args.n, args.seed, tok=lm.tok,
                                        target_tokens=args.target_prompt,
                                        max_shots=args.max_shots,
                                        offset=args.offset, total=args.total,
                                        which=which))
        names = [t._task_name for t in all_tasks]
    else:
        which = ("gsm8k", "math500")
        if args.task != "both":
            which = (args.task,)
        all_tasks = list(build_tasks(args.n, args.seed, args.nshot,
                                     target_tokens=args.target_prompt,
                                     tok=lm.tok, max_shots=args.max_shots,
                                     offset=args.offset, total=args.total,
                                     which=which))
        names = [t._task_name for t in all_tasks]
    if args.gen_toks:
        # Ask the task to request exactly this many tokens, and cap the engine
        # too, so the two can never disagree.
        for t in all_tasks:
            t.gen_toks = int(args.gen_toks)
        lm.max_gen_toks = int(args.gen_toks)
    nshot = int(all_tasks[0].config["num_fewshot"] or 0)
    task_dict = dict(zip(names, all_tasks))
    t0 = time.time()
    # ``evaluate`` (not ``simple_evaluate``): the latter routes through
    # TaskManager.load(), which only accepts task *names* / YAML configs and
    # chokes on hand-built Task instances.
    res = lm_eval.evaluate(
        lm=lm, task_dict=task_dict, system_instruction=SYS_TURN,
        log_samples=True, bootstrap_iters=100)

    loop_rets = lm.stats.get("loop_ret") or []
    per_loop = ({k: round(sum(r.get(k, 1.0) for r in loop_rets) / len(loop_rets), 4)
                 for k in loop_rets[0]} if loop_rets else None)

    out = {
        "harness": "lm-eval==0.4.13 (EleutherAI LM Harness)",
        "mode": args.mode,
        "config": lm.cfg.describe(),
        "model": args.model, "n": args.n, "nshot": nshot,
        "seed": args.seed, "target_prompt_tokens": args.target_prompt,
        "gen_toks": args.gen_toks or 256, "batch": args.batch,
        "max_gen_toks": args.max_gen_toks,
        "per_loop_retention": per_loop,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "kv_MiB_per_sample": round(lm.stats["kv_bytes"] / max(lm.stats["n"], 1)
                                   / 1024 ** 2, 2),
        "decode_ms_per_step": round(1000 * lm.stats["decode_s"]
                                    / max(lm.stats["decode_steps"], 1), 1),
        "results": {},
        "samples": {},
    }
    for name in res["results"]:
        out["results"][name] = res["results"][name]
    for name in res.get("samples", {}):
        out["samples"][name] = [
            {"gold": s["doc"]["target"], "ok": bool(s["exact_match"]),
             "out": (s.get("filtered_resps") or s.get("resps") or [""])[0][:400]
             if isinstance((s.get("filtered_resps") or s.get("resps") or [""])[0], str)
             else ""}
            for s in res["samples"][name]]
    tag = f"{args.mode}_rho{args.rho}_K{args.keep_loops}" if args.mode != "full" else "fullkv"
    path = os.path.join(ROOT, "results",
                        f"lmeval_{tag}_n{args.n}_s{args.seed}_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    print("saved ->", path)


if __name__ == "__main__":
    main()
