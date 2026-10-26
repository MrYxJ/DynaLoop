"""Evaluation utilities shipped with DynaLoop (no extra deps beyond pandas).

``datasets``  : offline loaders + answer graders for GSM8K / MATH-500 /
                ARC-Challenge / HellaSwag, and the few-shot prompt builders.
``niah``      : synthetic needle-in-a-haystack probes (single and multi-key).
``harness``   : ``lm_eval`` glue -- only imported when you actually use it
                (``lm_eval`` is an optional dependency).
"""

from __future__ import annotations

__all__ = ["datasets", "niah"]
