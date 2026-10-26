"""Needle-in-a-haystack probes used to test long-context retention.

Two generators:

* :func:`build_niah` -- one key/value needle at a controlled **depth**.
  Ouro-1.4B solves this at 100 % even with 0.37 retention, so it cannot
  discriminate between policies;
* :func:`build_niah_multi` -- ``n`` different key/value pairs hidden in the
  same haystack, then one of them is queried.  The model must keep *several*
  needles and pick the right one, which is what actually breaks when too many
  prompt tokens are evicted.  This is the default probe.

Both are fully synthetic: no download, deterministic given ``seed``.
"""

from __future__ import annotations

import random

import torch

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")

WORDS = ["cedar", "maple", "birch", "willow", "aspen", "oak", "pine",
         "elm", "fir", "spruce", "alder", "poplar"]
GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
         "theta", "iota", "kappa", "lambda", "mu"]


def build_niah(context_tokens: int, depth: float, seed: int) -> dict:
    """Single needle: ``{context, question, answer}``."""
    rng = random.Random(seed)
    key = "hollow-cedar"
    value = str(rng.randint(1000000, 9999999))
    needle = f"One of the special magic numbers for {key} is: {value}."
    # crude token accounting: ~4 chars per token
    n_filler = max(int(context_tokens * 4 / len(FILLER)) + 1, 8)
    parts = [FILLER] * n_filler
    parts.insert(int(len(parts) * depth), needle + " ")
    return {
        "context": "".join(parts),
        "question": f"What is the special magic number for {key} mentioned in the text?",
        "answer": value,
    }


def build_niah_multi(context_tokens: int, n_needles: int, seed: int) -> dict:
    """Multi-key needle probe: ``{context, question, answer}``."""
    rng = random.Random(seed)
    k = min(n_needles, len(WORDS))
    keys = [f"{a}-{b}" for a, b in
            zip(random.Random(seed).sample(WORDS, k=k), GREEK[:k])]
    pairs = {kk: str(rng.randint(1000000, 9999999)) for kk in keys}
    needles = [f"One of the special magic numbers for {kk} is: {v}."
               for kk, v in pairs.items()]
    n_filler = max(int(context_tokens * 4 / len(FILLER)) + 1, 8)
    parts = [FILLER] * n_filler
    slots = sorted(rng.sample(range(len(parts)), k=min(len(needles), len(parts))))
    for pos, needle in zip(slots, needles):
        parts.insert(pos, needle + " ")
    qkey = keys[rng.randrange(len(keys))]
    return {
        "context": "".join(parts),
        "question": f"What is the special magic number for {qkey}?",
        "answer": pairs[qkey],
    }


def niah_prompt(item: dict) -> str:
    """The plain completion-style prompt used by the NIAH runner."""
    return f"Question: {item['context']}\n\n{item['question']}\nAnswer:"


def encode(tok, text: str) -> torch.Tensor:
    """``tok(...)`` returns a BatchEncoding in transformers 4.x and 5.x alike."""
    out = tok(text, return_tensors="pt")
    if not isinstance(out, torch.Tensor):
        out = out["input_ids"]
    return out
