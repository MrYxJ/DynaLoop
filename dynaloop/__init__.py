"""DynaLoop -- dynamic, adaptive KV-cache budgeting for **looped** LLMs.

Package map
-----------
``dynaloop.policy``
    The allocator.  Turns per-(loop, layer) attention divergence into a
    per-(loop, layer) KV budget.  Pure tensor code, GPU-optional, unit-testable.
``dynaloop.loopkv``
    Hand-written attention against an explicit KV bank + RoPE helpers.
``dynaloop.engine``
    Single-sample looped forward pass (``OuroLoopEngine``).  Simple, readable,
    good for debugging and for inspecting budgets.
``dynaloop.batched_engine``
    Batched looped forward pass (``BatchedOuroEngine``).  Same policy, much
    faster: one weight read serves ``B`` samples.
``dynaloop.compat``
    Loading Ouro under transformers >= 5.x.
``dynaloop.eval``
    Datasets (GSM8K / MATH-500 / ARC-C / HellaSwag), graders, NIAH probes and
    optional ``lm_eval`` glue.

Quick start
-----------
>>> from dynaloop import DynaLoopConfig, BatchedOuroEngine, load_ouro
>>> model, tok, _ = load_ouro("Ouro-1.4B")
>>> cfg = DynaLoopConfig(allocator="js", keep_loops=1, js_kappa=0.5,
...                      js_gamma=1.0, js_floor=0.10, chunk_size=5)
>>> eng = BatchedOuroEngine(model, tok, cfg)

See ``README.md`` for the full story and ``examples/`` for runnable scripts.
"""

from __future__ import annotations

from .policy import (DynaLoopConfig, allocate_block_budgets, allocate_js_budgets,
                     chunk_scores, js_divergence, keep_indices_for_unit,
                     window_query_scores)

__version__ = "0.1.0"

__all__ = [
    "DynaLoopConfig",
    "allocate_js_budgets",
    "allocate_block_budgets",
    "keep_indices_for_unit",
    "window_query_scores",
    "chunk_scores",
    "js_divergence",
    "__version__",
]


def __getattr__(name):
    """Lazy imports for the torch/model-heavy symbols.

    Importing ``dynaloop`` must stay cheap (and must work without a GPU), so
    the engine and loader are only imported when asked for.
    """
    if name in ("OuroLoopEngine",):
        from .engine import OuroLoopEngine
        return OuroLoopEngine
    if name in ("BatchedOuroEngine",):
        from .batched_engine import BatchedOuroEngine
        return BatchedOuroEngine
    if name in ("load_ouro",):
        from .compat import load_ouro
        return load_ouro
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
