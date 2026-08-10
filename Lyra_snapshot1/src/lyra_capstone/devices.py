"""Device selection.

DEVIATION FROM SPEC §0.3. The spec assumes two GPUs (48 GB for training,
12 GB for embedding / judging, running concurrently). This box has a single
RTX A6000 (48 GB). Every `cuda:1` in the spec's signatures therefore collapses
to the one device, and the workloads the spec ran in parallel are sequenced
instead:

  - Phase 4 index builds no longer overlap Phase 3 training
  - The NLI judge (Phase 9) runs after generation (Phase 8), not beside it

Neither costs much: the parallelizable stages are the cheap ones. Nothing else
in the plan depends on the second card.

Every script resolves its device through here so the assumption lives in one
file if a second GPU ever appears.
"""

from __future__ import annotations

import os

TRAIN_DEVICE_ENV = "LYRA_TRAIN_DEVICE"
INFER_DEVICE_ENV = "LYRA_INFER_DEVICE"


def _default() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        # Local-laptop deviation from the spec's CUDA-only assumption above:
        # Apple Silicon has no VRAM counter equivalent to torch.cuda's, so
        # peak_vram_gb() below stays 0.0 on this path rather than reporting a
        # number that isn't comparable to the spec's GB figures.
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def train_device() -> str:
    return os.environ.get(TRAIN_DEVICE_ENV) or _default()


def infer_device() -> str:
    """Embedding, reranking, and NLI judging. Same card here; see module docstring."""
    return os.environ.get(INFER_DEVICE_ENV) or _default()


def sdpa_model_kwargs(device: str) -> dict:
    """Local-laptop deviation: MPS's scaled_dot_product_attention kernel does
    not implement dropout (PyTorch 2.13), which XLM-RoBERTa's attention uses
    during training. Forcing eager attention avoids that kernel on MPS only;
    CUDA keeps the default (faster) SDPA path."""
    return {"attn_implementation": "eager"} if device == "mps" else {}


def describe() -> dict:
    info = {"train": train_device(), "infer": infer_device(), "gpus": []}
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_mem_gb": round(props.total_memory / 1024**3, 1),
                }
            )
    except ImportError:
        pass
    return info


def peak_vram_gb(device: str | None = None) -> float:
    """Peak allocated VRAM since the last reset — logged in training manifests
    for the reproducibility appendix (Phase 3 acceptance criteria)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
    except ImportError:
        return 0.0
