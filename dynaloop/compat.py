"""Compatibility shim: load Ouro under transformers >= 5.x.

Ouro's remote code was written for transformers 4.55/4.56 and reads
``config.pad_token_id``, which transformers 5.x no longer defines by default
(RECON.md §9 conflict #3).  We patch the attribute before instantiation
instead of touching the vendored remote code, so the model files stay
byte-identical to the upstream repo.

Any further 5.x incompatibility surfaces here as well and is reported, never
silently worked around in the model code.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_ouro(model_path: str, device: str = "cuda",
              dtype: Optional[torch.dtype] = torch.bfloat16):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # transformers 5.x dropped the pad_token_id default that Ouro's code reads.
    if not hasattr(cfg, "pad_token_id"):
        try:
            cfg.pad_token_id = getattr(cfg, "eos_token_id", 0)
        except Exception:  # noqa: BLE001 - read-only config objects
            object.__setattr__(cfg, "pad_token_id", getattr(cfg, "eos_token_id", 0))

    # transformers 5.x dropped the "default" rope type from ROPE_INIT_FUNCTIONS
    # (registry now: dynamic/linear/llama3/longrope/proportional/yarn).  Ouro's
    # code asks for "default"; linear with factor 1.0 is numerically identical
    # (inv_freq divided by factor), so we express the same RoPE that way.
    # NB: transformers 5.x *normalises* ``rope_scaling: null`` into
    # ``{"rope_type": "default", ...}``, so both shapes must be handled.
    rs = getattr(cfg, "rope_scaling", None)
    rope_type = rs.get("rope_type") if isinstance(rs, dict) else rs
    if rope_type in (None, "default", ""):
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if "default" not in ROPE_INIT_FUNCTIONS:
                cfg.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        except Exception:  # noqa: BLE001
            pass

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=dtype, device_map=device,
        trust_remote_code=True)
    model.eval()
    # our engine always uses the eager path so attention weights/scores are
    # computed exactly as in modeling_ouro.eager_attention_forward
    try:
        model.config._attn_implementation = "eager"
    except Exception:  # noqa: BLE001
        pass
    return model, tok, cfg
