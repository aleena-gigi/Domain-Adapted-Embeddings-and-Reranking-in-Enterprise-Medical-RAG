"""Dataclass config loader (spec detail §0.1).

Configs live in YAML under configs/ and are loaded into typed dataclasses so
a typo becomes a TypeError at load time rather than a silently-defaulted
hyperparameter three hours into a run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

from . import paths
from .models.biencoder import BiEncoderConfig

T = TypeVar("T")


# --------------------------------------------------------------------------
# Phase 0-2 — data
# --------------------------------------------------------------------------


@dataclass
class MedEmbedExpectations:
    """Ground truth from spec v2 §2.1. Phase 0 asserts against these."""

    dataset_id: str = "abhinand/MedEmbed-training-triplets-v1"
    triplets: int = 232_684
    queries: int = 21_689
    corpus: int = 27_590


@dataclass
class NFCorpusExpectations:
    """Spec v2 §5.2 tier 2."""

    dataset_id: str = "BeIR/nfcorpus"
    docs: int = 3_633
    test_queries: int = 323


@dataclass
class AnaphoricFilterConfig:
    """D5 — the load-bearing filter."""

    enabled: bool = True
    # Path to the clinical vocabulary term list. Open Item #2 resolved toward a
    # MeSH/UMLS term dump over scispacy, on install friction (spec §12).
    vocab_path: str = "artifacts/interim/clinical_vocab.txt"
    min_vocab_term_len: int = 3
    validation_sample_size: int = 100
    validation_seed: int = 42


@dataclass
class FalseNegativeConfig:
    """Issue 2. The threshold is set from the histogram, never a priori (§12 item 4)."""

    enabled: bool = True
    scorer_model: str = "BAAI/bge-m3"
    # null until the distribution has been inspected; audit reports percentiles
    # and the chosen value is written back here.
    sim_threshold: float | None = None
    high_reuse_percentile: float = 99.0
    encode_batch_size: int = 64


@dataclass
class DegeneratePositiveConfig:
    """Issue 3."""

    enabled: bool = True
    min_tokens: int = 8


@dataclass
class SplitConfig:
    """Phase 2. Ratios and negative count live on DataConfig — the spec puts
    them there — so this holds only what splitting itself needs."""

    # Stock encoder, used both for near-duplicate detection and re-mining.
    # Never the fine-tune: Phase 2 runs before Phase 3 exists.
    scorer_model: str = "BAAI/bge-m3"
    encode_batch_size: int = 64
    # Cosine above which an eval query counts as a near-duplicate of a train
    # query. null on the first run: report the distribution, flag nothing, then
    # set the value from the observed pairs (same discipline as Open Item #4).
    semantic_dup_threshold: float | None = None
    dup_chunk_size: int = 512
    mine_chunk_size: int = 256
    # Cap on re-mined negative similarity. null inherits
    # false_negatives.sim_threshold, which is what you want: top-k mining aims
    # straight at the region where false negatives live.
    mine_max_similarity: float | None = None


@dataclass
class DataConfig:
    seed: int = 42
    medembed: MedEmbedExpectations = field(default_factory=MedEmbedExpectations)
    nfcorpus: NFCorpusExpectations = field(default_factory=NFCorpusExpectations)
    anaphoric: AnaphoricFilterConfig = field(default_factory=AnaphoricFilterConfig)
    false_negatives: FalseNegativeConfig = field(default_factory=FalseNegativeConfig)
    degenerate_positives: DegeneratePositiveConfig = field(
        default_factory=DegeneratePositiveConfig
    )
    splits: SplitConfig = field(default_factory=SplitConfig)
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    n_negatives_per_query: int = 8
    # Phase 1 acceptance criterion: below this, stop and escalate rather than
    # training on a corpus the filters have gutted.
    min_surviving_queries: int = 8_000
    figure_dpi: int = 300


# --------------------------------------------------------------------------
# Phase 3 — Experiment A
# --------------------------------------------------------------------------


@dataclass
class SweepGrid:
    """The spec's suggested grid (§3): 2 x 3 x 2 = 12 runs. An epoch is a few
    hundred steps at this dataset size, which is what makes a real sweep — and
    therefore the paper's comparative-optimization section — affordable."""

    learning_rate: list = field(default_factory=lambda: [1e-5, 2e-5])
    epochs: list = field(default_factory=lambda: [1, 2, 3])
    n_hard_negatives: list = field(default_factory=lambda: [4, 8])


@dataclass
class TrainAConfig:
    seed: int = 42
    figure_dpi: int = 300
    # BiEncoderConfig lives in models/biencoder.py because that is where the
    # spec puts it; the loader resolves nested dataclasses out of this module's
    # namespace, so it has to be imported by name here.
    model: BiEncoderConfig = field(default_factory=BiEncoderConfig)
    sweep: SweepGrid = field(default_factory=SweepGrid)


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def _build(cls: Type[T], data: Any) -> T:
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping for {cls.__name__}, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise TypeError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = known[name].type
        # Resolve string annotations from `from __future__ import annotations`.
        if isinstance(ftype, str):
            ftype = globals().get(ftype.split("|")[0].strip(), None)
        if is_dataclass(ftype):
            kwargs[name] = _build(ftype, value)
        elif isinstance(value, list) and name == "split_ratios":
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(cls: Type[T], path: str | Path | None = None) -> T:
    """Load `path` (default configs/<name>.yaml) into dataclass `cls`."""
    if path is None:
        stem = cls.__name__.replace("Config", "").lower()
        path = paths.CONFIGS / f"{stem}.yaml"
    path = Path(path)
    if not path.exists():
        return cls()  # defaults are the spec's values; a missing file is not fatal
    raw = yaml.safe_load(path.read_text()) or {}
    return _build(cls, raw)


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg) if is_dataclass(cfg) else dict(cfg)
