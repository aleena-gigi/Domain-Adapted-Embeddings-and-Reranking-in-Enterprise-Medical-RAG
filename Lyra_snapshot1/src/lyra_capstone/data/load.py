"""Phase 0 — data acquisition.

Pull source data once, immutably. Nothing here transforms anything: the raw
tree is the reproducibility anchor for every downstream number, so it is
written once, hashed, and made read-only.
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

import pandas as pd

from ..config import MedEmbedExpectations, NFCorpusExpectations

log = logging.getLogger(__name__)

MEDEMBED_CONFIGS = ("merged", "corpus", "queries")


class DatasetIntegrityError(RuntimeError):
    """Upstream row counts disagree with the spec. Never downgrade to a warning:
    a silent revision change invalidates every figure in the paper."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _repo_revision(repo_id: str) -> str:
    """Resolved commit SHA of the dataset repo — the manifest records this,
    not just the name (Phase 0 acceptance criteria)."""
    try:
        from huggingface_hub import dataset_info

        return dataset_info(repo_id).sha or "unknown"
    except Exception as exc:  # network/auth/hub-version differences
        log.warning("could not resolve revision for %s: %s", repo_id, exc)
        return "unknown"


def _expected_rows(repo_id: str, config: str) -> int | None:
    """Row count from dataset metadata, without materializing the split."""
    from datasets import load_dataset_builder

    builder = load_dataset_builder(repo_id, config)
    splits = builder.info.splits or {}
    return sum(s.num_examples for s in splits.values()) or None


def _assert_count(label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise DatasetIntegrityError(
            f"{label}: expected {expected:,} rows, got {actual:,}. "
            "The upstream dataset revision has changed; re-validate the spec's "
            "§2.1 figures before proceeding."
        )
    log.info("%s: %s rows OK", label, f"{actual:,}")


def set_readonly(path: Path, readonly: bool = True) -> None:
    """chmod -R a-w (or restore write) over the raw tree."""
    for p in [path, *path.rglob("*")]:
        mode = p.stat().st_mode
        if readonly:
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        else:
            p.chmod(mode | stat.S_IWUSR)


# --------------------------------------------------------------------------
# MedEmbed
# --------------------------------------------------------------------------


def download_medembed(
    dest: Path, expect: MedEmbedExpectations | None = None, force: bool = False
) -> dict:
    """Pull abhinand/MedEmbed-training-triplets-v1 from HF.

    Fetches the three configs the project uses: 'merged' (21,689 rows, all
    negatives per query — this is the training view), 'corpus' (27,590
    passages) and 'queries' (21,689). The 'default' config (232,684 flat
    triplets) is the same data exploded one-negative-per-row; its count is
    asserted from metadata rather than downloaded, since 'merged' already
    carries every negative.

    Persists as parquet. Does NOT modify.
    """
    from datasets import load_dataset

    expect = expect or MedEmbedExpectations()
    dest = Path(dest)
    if dest.exists() and not force:
        set_readonly(dest, readonly=False)
    dest.mkdir(parents=True, exist_ok=True)

    revision = _repo_revision(expect.dataset_id)
    log.info("MedEmbed revision: %s", revision)

    counts: dict[str, int] = {}
    for config in MEDEMBED_CONFIGS:
        ds = load_dataset(expect.dataset_id, config, split="train")
        out = dest / f"{config}.parquet"
        ds.to_parquet(out)
        counts[config] = ds.num_rows
        log.info("wrote %s (%s rows)", out.name, f"{ds.num_rows:,}")

    # 'default' is metadata-only: exploded view of the same triplets.
    default_rows = _expected_rows(expect.dataset_id, "default")
    counts["default"] = default_rows or -1

    _assert_count("MedEmbed merged (unique queries)", counts["merged"], expect.queries)
    _assert_count("MedEmbed queries", counts["queries"], expect.queries)
    _assert_count("MedEmbed corpus (passages)", counts["corpus"], expect.corpus)
    if default_rows is not None:
        _assert_count("MedEmbed default (triplets)", default_rows, expect.triplets)

    return {
        "dataset_id": expect.dataset_id,
        "revision": revision,
        "counts": counts,
        "path": str(dest),
    }


# --------------------------------------------------------------------------
# NFCorpus (tier 2, out-of-domain transfer check)
# --------------------------------------------------------------------------


def download_nfcorpus(
    dest: Path, expect: NFCorpusExpectations | None = None, force: bool = False
) -> dict:
    """Pull NFCorpus from BEIR. Expect 3,633 docs / 323 test queries.

    Qrels carry graded relevance (0-2). Grades are preserved — nDCG needs
    them; they are NOT binarized here (spec detail §7, ndcg_at_k).

    The BeIR/nfcorpus 'queries' config holds all 3,237 queries across splits;
    the test set is defined by the distinct query ids in the test qrels, which
    is where the 323 comes from.
    """
    from datasets import load_dataset

    expect = expect or NFCorpusExpectations()
    dest = Path(dest)
    if dest.exists() and not force:
        set_readonly(dest, readonly=False)
    dest.mkdir(parents=True, exist_ok=True)

    qrels_id = f"{expect.dataset_id}-qrels"
    revision = _repo_revision(expect.dataset_id)
    qrels_revision = _repo_revision(qrels_id)

    corpus = load_dataset(expect.dataset_id, "corpus", split="corpus")
    queries = load_dataset(expect.dataset_id, "queries", split="queries")
    qrels = load_dataset(qrels_id, split="test")

    corpus.to_parquet(dest / "corpus.parquet")
    queries.to_parquet(dest / "queries.parquet")
    qrels.to_parquet(dest / "qrels_test.parquet")

    qrels_df = qrels.to_pandas()
    n_test_queries = qrels_df["query-id"].nunique()
    grade_counts = qrels_df["score"].value_counts().sort_index().to_dict()

    _assert_count("NFCorpus corpus", corpus.num_rows, expect.docs)
    _assert_count("NFCorpus test queries", int(n_test_queries), expect.test_queries)

    if set(grade_counts) - {0, 1, 2}:
        log.warning("unexpected NFCorpus relevance grades: %s", sorted(grade_counts))

    return {
        "dataset_id": expect.dataset_id,
        "revision": revision,
        "qrels_dataset_id": qrels_id,
        "qrels_revision": qrels_revision,
        "counts": {
            "corpus": corpus.num_rows,
            "queries_all_splits": queries.num_rows,
            "test_queries": int(n_test_queries),
            "qrels_test_rows": len(qrels_df),
        },
        "relevance_grades": {int(k): int(v) for k, v in grade_counts.items()},
        "path": str(dest),
    }


# --------------------------------------------------------------------------
# readers used by later phases
# --------------------------------------------------------------------------


def load_medembed_merged(raw_dir: Path) -> pd.DataFrame:
    """Per-query view: query, query_id, pos, pos_id, neg (list), neg_id (list)."""
    return pd.read_parquet(Path(raw_dir) / "merged.parquet")


def load_medembed_corpus(raw_dir: Path) -> pd.DataFrame:
    """Passage store: id, text."""
    return pd.read_parquet(Path(raw_dir) / "corpus.parquet")


def corpus_as_dict(raw_dir: Path) -> dict[str, str]:
    df = load_medembed_corpus(raw_dir)
    return dict(zip(df["id"], df["text"]))
