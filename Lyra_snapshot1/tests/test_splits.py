"""Phase 2 leakage assertions, executable outside the notebook.

Spec detail §2 requires the leakage audit to live somewhere that runs on every
build rather than in a notebook cell a reader has to trust. There is no CI
runner on this project (see docs/decisions.md §E1), so the assertion lives in
two places: inside `scripts/02_build_splits.py`, which raises and aborts, and
here, where `pytest` re-derives it from the written artifacts.

    .venv/bin/python -m pytest tests/ -q

The synthetic tests always run. The artifact tests skip when
artifacts/processed/ has not been built, so a fresh checkout is still green.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyra_capstone import paths  # noqa: E402
from lyra_capstone.data import splits  # noqa: E402


# --------------------------------------------------------------------------
# synthetic — the invariants themselves
# --------------------------------------------------------------------------


def _toy_frame(rows):
    return pd.DataFrame(
        [{"query_id": q, "query": t, "pos_id": p, "pos": "x", "neg_id": n, "neg": []}
         for q, t, p, n in rows]
    )


def test_partition_is_disjoint_and_total():
    ids = [f"p{i}" for i in range(1000)]
    parts = splits.partition_corpus(ids, (0.8, 0.1, 0.1), seed=42)
    assert sum(len(v) for v in parts.values()) == 1000
    assert not parts["train"] & parts["val"]
    assert not parts["train"] & parts["test"]
    assert not parts["val"] & parts["test"]
    assert len(parts["train"]) == 800


def test_partition_is_seed_stable_and_order_independent():
    ids = [f"p{i}" for i in range(500)]
    a = splits.partition_corpus(ids, seed=42)
    b = splits.partition_corpus(list(reversed(ids)), seed=42)
    c = splits.partition_corpus(ids, seed=7)
    assert a == b, "partition must not depend on input order"
    assert a != c, "different seeds must give different partitions"


def test_query_inherits_its_positive_partition():
    parts = {"train": {"p1"}, "val": {"p2"}, "test": {"p3"}}
    df = _toy_frame([("q1", "a", "p1", []), ("q2", "b", "p3", [])])
    out = splits.assign_queries_to_splits(df, parts)
    assert list(out["split"]) == ["train", "test"]


def test_unknown_positive_raises():
    parts = {"train": {"p1"}, "val": set(), "test": set()}
    df = _toy_frame([("q1", "a", "p_missing", [])])
    with pytest.raises(splits.LeakageError):
        splits.assign_queries_to_splits(df, parts)


def test_audit_raises_when_a_train_negative_points_at_a_test_passage():
    parts = {"train": {"p1"}, "val": {"p2"}, "test": {"p3"}}
    train = _toy_frame([("q1", "a", "p1", ["p3"])])  # <- the leak
    val = _toy_frame([("q2", "b", "p2", [])])
    test = _toy_frame([("q3", "c", "p3", [])])
    with pytest.raises(splits.LeakageError, match="test_passages_seen_in_train"):
        splits.audit_leakage(train, val, test, parts)


def test_audit_raises_on_a_query_straddling_splits():
    parts = {"train": {"p1"}, "val": {"p2"}, "test": {"p3"}}
    train = _toy_frame([("q1", "Sepsis management?", "p1", [])])
    val = _toy_frame([("q2", "x", "p2", [])])
    test = _toy_frame([("q3", "sepsis management!", "p3", [])])  # normalizes equal
    with pytest.raises(splits.LeakageError, match="exact_duplicate_queries"):
        splits.audit_leakage(train, val, test, parts)


def test_duplicate_resolution_keeps_the_higher_precedence_split():
    parts = {"train": {"p1"}, "val": {"p2"}, "test": {"p3"}}
    df = splits.assign_queries_to_splits(
        _toy_frame([
            ("q_train", "Sepsis management?", "p1", []),
            ("q_val", "sepsis management", "p2", []),
            ("q_test", "SEPSIS MANAGEMENT!", "p3", []),
        ]),
        parts,
    )
    out, stats = splits.resolve_cross_split_duplicates(df, pd.DataFrame())
    assert list(out["query_id"]) == ["q_test"], "test copy must be the survivor"
    assert stats["queries_dropped_exact"] == 2


# --------------------------------------------------------------------------
# artifacts — the real splits, re-audited from disk
# --------------------------------------------------------------------------


def _load(name: str) -> pd.DataFrame:
    path = paths.PROCESSED / f"{name}.jsonl"
    if not path.exists():
        pytest.skip("artifacts/processed not built — run scripts/02_build_splits.py")
    return pd.read_json(path, lines=True, dtype=False)


@pytest.fixture(scope="module")
def built():
    corpus_path = paths.PROCESSED / "corpus.jsonl"
    if not corpus_path.exists():
        pytest.skip("artifacts/processed not built — run scripts/02_build_splits.py")
    corpus = pd.read_json(corpus_path, lines=True, dtype=False)
    partitions = {
        name: set(corpus.loc[corpus["split"] == name, "id"])
        for name in splits.SPLIT_NAMES
    }
    return _load("train"), _load("val"), _load("test"), partitions


def test_written_splits_pass_the_leakage_audit(built):
    train, val, test, partitions = built
    result = splits.audit_leakage(train, val, test, partitions)
    assert result["n_violations"] == 0


def test_written_splits_match_the_recorded_manifest(built):
    train, val, test, _ = built
    recorded = json.loads((paths.PROCESSED / "split_manifest.json").read_text())
    sizes = recorded["leakage_audit"]["sizes"]
    assert (len(train), len(val), len(test)) == (
        sizes["train_queries"], sizes["val_queries"], sizes["test_queries"]
    )


def test_train_rows_are_well_formed(built):
    train, _, _, partitions = built
    assert (train["neg_id"].apply(len) == train["neg"].apply(len)).all()
    assert (train["neg_id"].apply(lambda n: len(set(n)) == len(n))).all()
    assert not any(p in set(n) for p, n in zip(train["pos_id"], train["neg_id"]))
    assert all(nid in partitions["train"] for negs in train["neg_id"] for nid in negs)


def test_full_corpus_is_written_for_the_eval_index(built):
    corpus = pd.read_json(paths.PROCESSED / "corpus.jsonl", lines=True, dtype=False)
    # Every configuration indexes all 27,590 passages so the distractor pool is
    # identical across the results table; the split governs training exposure,
    # not index membership.
    assert len(corpus) == 27_590
    assert corpus["id"].is_unique
