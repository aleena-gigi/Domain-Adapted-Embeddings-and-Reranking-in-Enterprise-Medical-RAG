"""Global determinism (spec detail §0.4)."""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, strict: bool = False) -> int:
    """Seed random / numpy / torch. Returns the seed so callers can log it.

    `strict` turns on torch deterministic algorithms. It is off by default
    because it costs throughput in the contrastive training loop; the spec
    permits skipping it where it tanks throughput, but the choice is recorded
    in the manifest either way.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if strict:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
