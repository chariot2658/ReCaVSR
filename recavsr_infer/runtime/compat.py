"""Torch version compatibility for compile options."""

from __future__ import annotations

import torch


def inductor_options(options: dict) -> dict:
    """Drop Inductor options this torch build no longer knows.

    The release pins torch 2.10; newer builds removed `force_same_precision`
    (same-precision matmul templates became the only behavior).
    """
    known = torch._inductor.config.get_config_copy()
    return {k: v for k, v in options.items() if k in known}
