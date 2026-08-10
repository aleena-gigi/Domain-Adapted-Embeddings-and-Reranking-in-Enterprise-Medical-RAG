"""Artifact registry. No other module hardcodes a path (spec detail §0.2)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LYRA_ROOT", Path(__file__).resolve().parents[2]))

CONFIGS = ROOT / "configs"
ARTIFACTS = Path(os.environ.get("LYRA_ARTIFACTS", ROOT / "artifacts"))

RAW = ARTIFACTS / "raw"
INTERIM = ARTIFACTS / "interim"
PROCESSED = ARTIFACTS / "processed"
MODELS = ARTIFACTS / "models"
INDEXES = ARTIFACTS / "indexes"
RUNS = ARTIFACTS / "runs"
FIGURES = ARTIFACTS / "figures"

RAW_MEDEMBED = RAW / "medembed"
RAW_NFCORPUS = RAW / "nfcorpus"

BIENCODER_MODELS = MODELS / "biencoder"
RERANKER_MODELS = MODELS / "reranker"


def ensure_dirs() -> None:
    for p in (
        RAW,
        INTERIM,
        PROCESSED,
        MODELS,
        INDEXES,
        RUNS,
        FIGURES,
        BIENCODER_MODELS,
        RERANKER_MODELS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def index_dir(config_name: str, tier: str) -> Path:
    """artifacts/indexes/{config_name}/{tier}/"""
    return INDEXES / config_name / tier


def run_dir(run_id: str) -> Path:
    """artifacts/runs/{run_id}/"""
    return RUNS / run_id


def model_dir(kind: str, run_id: str) -> Path:
    """artifacts/models/{biencoder|reranker}/{run_id}/"""
    if kind not in ("biencoder", "reranker"):
        raise ValueError(f"unknown model kind: {kind}")
    return MODELS / kind / run_id


def figure(name: str) -> Path:
    """artifacts/figures/{name} — figures are written at 300 dpi (§1.4)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name
