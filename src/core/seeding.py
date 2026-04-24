"""Single-knob reproducibility seeding for numpy + random + torch.

Shared by the experiment runner and the standalone-training dispatcher
so both paths seed identically. Torch is imported lazily — callers that
only need config validation shouldn't pay its ~4 s cold-import cost.
"""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed numpy + stdlib random + torch (if available) from one scalar."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
