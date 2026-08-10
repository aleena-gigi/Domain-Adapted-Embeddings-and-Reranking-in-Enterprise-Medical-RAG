"""Phase 2 — leak-free document-level splits (spec detail §2, spec v2 §3).

The partition is over **passages**, not queries. A query inherits the split of
its `pos_id`. That is the only construction under which a test passage can be
guaranteed unseen at training time, because the same passage frequently serves
several queries (18,768 distinct positives back 19,141 queries).

Three things have to hold when this module is done, and `audit_leakage` raises
rather than warns on any of them:

  1. no passage appears in two partitions;
  2. no train row references a val/test passage in *any* role — positive or
     negative (this is what `sanitize_training_negatives` buys);
  3. no query straddles splits, exactly or as a near-duplicate.

Rule 3 is the one that is easy to skip and expensive to discover later: two
near-identical queries pointing at different passages let the model memorise a
test query through its train twin without any passage ever leaking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


class LeakageError(AssertionError):
    """A split invariant was violated. Never downgrade this to a warning:
    silent leakage invalidates every downstream number in the paper."""


# --------------------------------------------------------------------------
# 2.1 Partition the corpus
# --------------------------------------------------------------------------


def partition_corpus(
    corpus_ids: list[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, set[str]]:
    """Partition the 27,590 PASSAGES — not the queries — 80/10/10.

    Deterministic in the ids, not in their arrival order: the list is sorted
    before permuting, so a reordered corpus file still yields the same
    partition. The acceptance criterion is byte-identical splits from the same
    seed, and sort-then-permute is what makes that hold across pandas versions.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} -> {sum(ratios)}")

    ids = sorted(set(corpus_ids))
    if len(ids) != len(corpus_ids):
        log.warning("corpus_ids contained %d duplicate ids", len(corpus_ids) - len(ids))

    perm = np.random.default_rng(seed).permutation(len(ids))
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    slices = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    parts = {name: {ids[i] for i in idx} for name, idx in slices.items()}

    if sum(len(v) for v in parts.values()) != n:
        raise LeakageError("partition lost or duplicated passages")
    log.info(
        "corpus partition: train=%s val=%s test=%s (of %s)",
        f"{len(parts['train']):,}", f"{len(parts['val']):,}",
        f"{len(parts['test']):,}", f"{n:,}",
    )
    return parts


def assign_queries_to_splits(
    df: pd.DataFrame, partitions: dict[str, set[str]]
) -> pd.DataFrame:
    """Each query inherits the split of its pos_id.

    Adds a `split` column. Raises if any pos_id is outside the corpus, which
    would mean Phase 1's output and Phase 0's corpus have drifted apart.
    """
    lookup: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for pid in partitions[name]:
            lookup[pid] = name

    out = df.copy()
    out["split"] = out["pos_id"].map(lookup)
    orphans = out["split"].isna()
    if orphans.any():
        sample = out.loc[orphans, "pos_id"].head(5).tolist()
        raise LeakageError(
            f"{int(orphans.sum())} queries have a pos_id outside the corpus "
            f"partition, e.g. {sample}"
        )

    counts = out["split"].value_counts().to_dict()
    log.info(
        "query splits: train=%s val=%s test=%s",
        f"{counts.get('train', 0):,}", f"{counts.get('val', 0):,}",
        f"{counts.get('test', 0):,}",
    )
    return out


# --------------------------------------------------------------------------
# 2.2 Cross-split duplicate queries
# --------------------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_query(q: str) -> str:
    """Case, punctuation and whitespace folded away — the exact-duplicate key."""
    return _NORM_RE.sub(" ", str(q).lower()).strip()


def find_exact_duplicate_queries(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized-exact duplicate queries, with the splits they straddle.

    Returns one row per duplicate *group*, not per query.
    """
    tmp = df[["query_id", "query", "split"]].copy()
    tmp["key"] = tmp["query"].map(normalize_query)
    grouped = tmp.groupby("key").agg(
        n=("query_id", "size"),
        splits=("split", lambda s: sorted(set(s))),
        query_ids=("query_id", list),
        example=("query", "first"),
    )
    dups = grouped[grouped["n"] > 1].reset_index()
    dups["straddles"] = dups["splits"].apply(len) > 1
    return dups


# Which copy of a straddling duplicate survives. Test is the reported number
# and is never touched; val is the model-selection instrument and outranks
# train. So a train/test twin costs the train copy, and a val/test twin costs
# the val copy. Reassigning a query to another split is not an option — it
# would break the "query inherits its pos_id's partition" invariant, which is
# the entire basis of the guarantee.
SPLIT_PRECEDENCE = {"test": 2, "val": 1, "train": 0}


def find_semantic_duplicate_queries(
    df: pd.DataFrame,
    threshold: float | None,
    model_name: str = "BAAI/bge-m3",
    device: str | None = None,
    batch_size: int = 64,
    chunk: int = 512,
    top_report: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """For every query, its nearest neighbour in a *different* split.

    All-pairs over 19k queries, same-split entries masked out, max only — the
    question is not "are there duplicates" but "does any query have a twin on
    the other side of the split boundary".

    Returns (pairs, distribution_stats). Every cross-split nearest-neighbour
    pair is returned, carrying a `flagged` column; with `threshold=None`
    nothing is flagged and only the distribution is reported, which is how the
    threshold gets chosen from the data rather than a priori. The full frame
    comes back either way so the bands can be inspected before committing to a
    value.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    from ..devices import infer_device

    device = device or infer_device()
    q = df.sort_values("query_id").reset_index(drop=True)
    if q["split"].nunique() < 2:
        return pd.DataFrame(), {"skipped": True}

    model = SentenceTransformer(model_name, device=device)
    log.info("encoding %s queries for the cross-split near-dup audit", f"{len(q):,}")
    emb = torch.from_numpy(model.encode(
        q["query"].tolist(), batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )).to(device)
    del model

    codes = torch.tensor(
        q["split"].map(SPLIT_PRECEDENCE).to_numpy(), device=device, dtype=torch.int16
    )
    best_sim = np.empty(len(q), dtype=np.float32)
    best_idx = np.empty(len(q), dtype=np.int64)
    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        sims = emb[start:stop] @ emb.T
        # Mask same-split candidates; self-similarity is masked with them.
        sims.masked_fill_(codes[start:stop, None] == codes[None, :], -2.0)
        vals, idx = sims.max(dim=1)
        best_sim[start:stop] = vals.float().cpu().numpy()
        best_idx[start:stop] = idx.cpu().numpy()

    del emb, codes
    torch.cuda.empty_cache()

    pairs = pd.DataFrame({
        "query_id": q["query_id"], "query": q["query"], "split": q["split"],
        "other_query_id": q["query_id"].to_numpy()[best_idx],
        "other_query": q["query"].to_numpy()[best_idx],
        "other_split": q["split"].to_numpy()[best_idx],
        "similarity": best_sim,
    })
    # Each twin is found from both sides; keep one row per unordered pair.
    key = pairs.apply(
        lambda r: tuple(sorted((r["query_id"], r["other_query_id"]))), axis=1
    )
    pairs = (
        pairs.assign(_key=key)
        .sort_values(["similarity", "query_id"], ascending=[False, True])
        .drop_duplicates("_key")
        .drop(columns="_key")
        .reset_index(drop=True)
    )

    qs = [50, 90, 95, 99, 99.5, 99.9]
    stats = {
        "n_pairs": int(len(pairs)),
        "mean": float(pairs["similarity"].mean()),
        "max": float(pairs["similarity"].max()),
        "percentiles": {
            str(p): float(np.percentile(pairs["similarity"], p)) for p in qs
        },
        # Concrete pairs, so the threshold is defensible in the write-up rather
        # than asserted.
        "top_pairs": pairs.head(top_report)[
            ["similarity", "split", "query", "other_split", "other_query"]
        ].to_dict("records"),
    }
    if threshold is None:
        pairs["flagged"] = False
        stats["threshold"] = None
        stats["n_flagged"] = 0
        log.warning(
            "splits.semantic_dup_threshold is null — reporting the near-dup "
            "distribution only, flagging nothing. Inspect top_pairs in "
            "split_manifest.json, set the threshold in configs/data.yaml, re-run."
        )
        return pairs, stats

    pairs["flagged"] = pairs["similarity"] >= threshold
    stats["threshold"] = float(threshold)
    stats["n_flagged"] = int(pairs["flagged"].sum())
    log.info("near-dup audit: %s cross-split query pairs >= %.3f",
             f"{stats['n_flagged']:,}", threshold)
    return pairs, stats


def _lower_precedence_id(a_id, a_split, b_id, b_split):
    """Of two straddling twins, the one that gets dropped."""
    if SPLIT_PRECEDENCE[a_split] < SPLIT_PRECEDENCE[b_split]:
        return a_id
    if SPLIT_PRECEDENCE[b_split] < SPLIT_PRECEDENCE[a_split]:
        return b_id
    return None  # same split — not a straddle, nothing to resolve


def resolve_cross_split_duplicates(
    df: pd.DataFrame, semantic_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Drop the lower-precedence copy of every cross-split duplicate.

    Exact duplicates resolve per group (keep only the members sitting in the
    highest-precedence split present); near-duplicates resolve per pair.
    """
    exact = find_exact_duplicate_queries(df)
    straddling = exact[exact["straddles"]]
    split_of = dict(zip(df["query_id"], df["split"]))

    drop_ids: set[str] = set()
    for _, row in straddling.iterrows():
        winner = max(row["splits"], key=lambda s: SPLIT_PRECEDENCE[s])
        drop_ids.update(
            qid for qid in row["query_ids"] if split_of[qid] != winner
        )
    n_exact = len(drop_ids)

    flagged_pairs = (
        semantic_pairs[semantic_pairs["flagged"]]
        if "flagged" in semantic_pairs.columns
        else semantic_pairs
    )
    for r in flagged_pairs.itertuples(index=False):
        loser = _lower_precedence_id(
            r.query_id, r.split, r.other_query_id, r.other_split
        )
        if loser is not None:
            drop_ids.add(loser)

    out = df[~df["query_id"].isin(drop_ids)].copy()
    dropped = df[df["query_id"].isin(drop_ids)]
    stats = {
        "exact_duplicate_groups": int(len(exact)),
        "exact_groups_straddling_splits": int(len(straddling)),
        "queries_dropped_exact": int(n_exact),
        "queries_dropped_semantic": int(len(drop_ids) - n_exact),
        "queries_dropped_total": int(len(drop_ids)),
        "queries_dropped_by_split": dropped["split"].value_counts().to_dict(),
        "n_before": int(len(df)),
        "n_after": int(len(out)),
    }
    if drop_ids:
        log.info("dropped %s cross-split duplicate queries %s",
                 f"{len(drop_ids):,}", stats["queries_dropped_by_split"])
    return out, stats


# --------------------------------------------------------------------------
# 2.3 Training-negative sanitation — the leakage guarantee
# --------------------------------------------------------------------------


@dataclass
class MiningStats:
    n_queries: int = 0
    negatives_before: int = 0
    negatives_dropped_out_of_partition: int = 0
    negatives_dropped_self: int = 0
    negatives_mined: int = 0
    queries_needing_mining: int = 0
    queries_short_after_mining: int = 0
    mined_candidates_rejected_too_similar: int = 0

    def as_dict(self) -> dict:
        return {k: int(v) for k, v in self.__dict__.items()}


def sanitize_training_negatives(
    train_df: pd.DataFrame,
    partitions: dict[str, set[str]],
    corpus: dict[str, str],
    n_negatives: int,
    device: str | None = None,
    model_name: str = "BAAI/bge-m3",
    max_similarity: float | None = None,
    batch_size: int = 64,
    chunk: int = 256,
) -> tuple[pd.DataFrame, dict]:
    """Remove any training negative pointing into val/test, then re-mine
    replacements from the TRAIN partition only using stock bge-m3 top-k.

    This is the leakage guarantee: after this call, no test-partition passage
    has been seen during training in any role, positive or negative.

    Two refinements the signature does not imply, both of which matter:

    * Mining is capped at `max_similarity` (Phase 1's false-negative threshold).
      Top-k against the query is *exactly* where false negatives live, so
      mining without the cap would reintroduce the pairs Phase 1 removed.
    * The result is padded **and truncated** to `n_negatives`, keeping the
      hardest survivors, so every training row has the same width. Phase 3
      wants a fixed (anchor, positive, negative_1..k) schema.

    Stock bge-m3, not the fine-tune: this runs before Phase 3 exists. Phase 4
    re-mines from the *fine-tuned* retriever, which is a different job.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    from ..devices import infer_device

    device = device or infer_device()
    train_ids = sorted(partitions["train"])
    train_id_set = partitions["train"]
    id_pos = {pid: i for i, pid in enumerate(train_ids)}

    df = train_df.sort_values("query_id").reset_index(drop=True)
    stats = MiningStats(n_queries=len(df))

    # ---- strip out-of-partition and self-referential negatives -----------
    kept_ids: list[list[str]] = []
    for negs, pos_id in zip(df["neg_id"], df["pos_id"]):
        negs = list(negs) if negs is not None else []
        stats.negatives_before += len(negs)
        keep, seen = [], set()
        for nid in negs:
            if nid == pos_id:
                stats.negatives_dropped_self += 1
                continue
            if nid not in train_id_set:
                stats.negatives_dropped_out_of_partition += 1
                continue
            if nid in seen:
                continue
            seen.add(nid)
            keep.append(nid)
        kept_ids.append(keep)

    log.info(
        "train negatives: %s dropped for pointing outside the train partition, "
        "%s self-referential",
        f"{stats.negatives_dropped_out_of_partition:,}",
        f"{stats.negatives_dropped_self:,}",
    )

    # ---- encode once, reuse for scoring and mining -----------------------
    model = SentenceTransformer(model_name, device=device)
    enc = dict(batch_size=batch_size, normalize_embeddings=True,
               show_progress_bar=True, convert_to_numpy=True)
    log.info("encoding %s train-partition passages", f"{len(train_ids):,}")
    p_emb = torch.from_numpy(model.encode([corpus[i] for i in train_ids], **enc)).to(device)
    log.info("encoding %s train queries", f"{len(df):,}")
    q_emb = torch.from_numpy(model.encode(df["query"].tolist(), **enc)).to(device)
    del model

    cap = float("inf") if max_similarity is None else float(max_similarity)
    # Headroom for the excluded set (positive, existing negatives, candidates
    # above the cap). At +32 a generic query can exhaust its top-k and finish
    # short; +120 costs nothing on a 22k-row topk and drives that to zero.
    top_k = min(n_negatives + 120, len(train_ids))

    final_ids: list[list[str]] = [None] * len(df)  # type: ignore[list-item]
    pos_ids = df["pos_id"].tolist()

    for start in range(0, len(df), chunk):
        stop = min(start + chunk, len(df))
        sims = q_emb[start:stop] @ p_emb.T
        # Never mine the row's own positive.
        for local, gid in enumerate(range(start, stop)):
            pi = id_pos.get(pos_ids[gid])
            if pi is not None:
                sims[local, pi] = -2.0
        vals, idx = torch.topk(sims, k=top_k, dim=1)
        vals_np = vals.float().cpu().numpy()
        idx_np = idx.cpu().numpy()
        sims_np = sims.float().cpu().numpy()

        for local, gid in enumerate(range(start, stop)):
            existing = kept_ids[gid]
            # Keep the hardest existing negatives first: they came from the
            # source dataset's own mining and are the ones Phase 1 vetted.
            if existing:
                order = np.argsort([-sims_np[local, id_pos[n]] for n in existing], kind="stable")
                existing = [existing[i] for i in order]
            chosen = existing[:n_negatives]
            if len(chosen) < n_negatives:
                stats.queries_needing_mining += 1
                taken = set(chosen)
                for rank in range(top_k):
                    if len(chosen) >= n_negatives:
                        break
                    cand = train_ids[idx_np[local, rank]]
                    if cand in taken:
                        continue
                    if vals_np[local, rank] >= cap:
                        stats.mined_candidates_rejected_too_similar += 1
                        continue
                    taken.add(cand)
                    chosen.append(cand)
                    stats.negatives_mined += 1
                if len(chosen) < n_negatives:
                    stats.queries_short_after_mining += 1
            final_ids[gid] = chosen

    del p_emb, q_emb
    torch.cuda.empty_cache()

    out = df.copy()
    out["neg_id"] = final_ids
    out["neg"] = [[corpus[i] for i in ids] for ids in final_ids]
    out["n_negatives"] = [len(ids) for ids in final_ids]

    d = stats.as_dict()
    d["mean_negatives_after"] = round(float(out["n_negatives"].mean()), 3)
    d["target_negatives_per_query"] = int(n_negatives)
    d["max_similarity_cap"] = None if max_similarity is None else float(max_similarity)
    log.info("train negatives after sanitation: mean %.3f per query", d["mean_negatives_after"])
    return out, d


# --------------------------------------------------------------------------
# 2.4 Serialization
# --------------------------------------------------------------------------

# Field order is fixed and rows are sorted by query_id so the same seed gives a
# byte-identical file — that is the Phase 2 reproducibility criterion, and
# dict iteration order is the easiest way to lose it.


def _as_list(value) -> list[str]:
    # neg_id/neg round-trip out of parquet as numpy arrays, whose truthiness
    # raises. Never write `value or []` here.
    if value is None:
        return []
    return [str(x) for x in value]


def query_records(df: pd.DataFrame) -> list[dict]:
    cols = ("query_id", "query", "split", "pos_id", "pos", "neg_id", "neg")
    ordered = df.sort_values("query_id")
    return [
        {
            "query_id": str(r[0]), "query": str(r[1]), "split": str(r[2]),
            "pos_id": str(r[3]), "pos": str(r[4]),
            "neg_id": _as_list(r[5]), "neg": _as_list(r[6]),
        }
        for r in ordered[list(cols)].itertuples(index=False, name=None)
    ]


def corpus_records(corpus_df: pd.DataFrame, partitions: dict[str, set[str]]) -> list[dict]:
    """Every passage with its partition label.

    All 27,590 are written, including the 8,822 that are never anyone's
    positive: at eval the full corpus is indexed for every configuration so the
    distractor pool is identical across the results table. The split governs
    training exposure, not index membership (spec detail §2).
    """
    lookup = {pid: name for name in SPLIT_NAMES for pid in partitions[name]}
    return [
        {"id": str(r.id), "text": str(r.text), "split": lookup[str(r.id)]}
        for r in corpus_df.sort_values("id").itertuples(index=False)
    ]


def write_jsonl(path, records: list[dict]):
    import json
    from pathlib import Path

    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("wrote %s rows -> %s", f"{len(records):,}", path)
    return path


# --------------------------------------------------------------------------
# 2.5 Leakage audit — assertions, not warnings
# --------------------------------------------------------------------------


def audit_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    partitions: dict[str, set[str]],
    extra_violations: dict[str, int] | None = None,
) -> dict:
    """Hard assertion pass. Must return zero violations across:
      - test pos_ids appearing anywhere in train
      - test/val pos_ids appearing as train negatives
      - duplicate queries straddling splits (exact + near-dup)
    Raise on any violation. Do not warn — raise.

    `extra_violations` carries counts from checks that need a GPU pass and so
    run in the driver script (semantic near-duplicates), keeping them inside
    the same all-or-nothing gate.
    """
    checks: dict[str, int] = {}

    # -- partitions themselves ---------------------------------------------
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        checks[f"partition_overlap_{a}_{b}"] = len(partitions[a] & partitions[b])

    # -- each query sits in its pos_id's partition --------------------------
    frames = {"train": train_df, "val": val_df, "test": test_df}
    for name, frame in frames.items():
        checks[f"{name}_pos_outside_own_partition"] = int(
            (~frame["pos_id"].isin(partitions[name])).sum()
        )

    # -- training exposure --------------------------------------------------
    train_negs: set[str] = set()
    for negs in train_df["neg_id"]:
        train_negs.update(_as_list(negs))
    train_pos = set(train_df["pos_id"])
    train_seen = train_pos | train_negs

    checks["test_passages_seen_in_train"] = len(partitions["test"] & train_seen)
    checks["val_passages_seen_in_train"] = len(partitions["val"] & train_seen)
    checks["test_pos_ids_in_train_any_role"] = len(set(test_df["pos_id"]) & train_seen)
    checks["val_pos_ids_as_train_negatives"] = len(set(val_df["pos_id"]) & train_negs)
    checks["test_pos_ids_as_train_negatives"] = len(set(test_df["pos_id"]) & train_negs)
    checks["train_negatives_equal_to_own_positive"] = int(
        sum(
            pid in set(_as_list(negs))
            for pid, negs in zip(train_df["pos_id"], train_df["neg_id"])
        )
    )

    # -- duplicate queries straddling splits --------------------------------
    combined = pd.concat(
        [f.assign(split=name)[["query_id", "query", "split"]] for name, f in frames.items()]
    )
    exact = find_exact_duplicate_queries(combined)
    checks["exact_duplicate_queries_across_splits"] = int(exact["straddles"].sum())
    checks["duplicate_query_ids_across_splits"] = int(
        len(combined) - combined["query_id"].nunique()
    )

    if extra_violations:
        checks.update({k: int(v) for k, v in extra_violations.items()})

    violations = {k: v for k, v in checks.items() if v}
    result = {
        "checks": checks,
        "violations": violations,
        "n_violations": len(violations),
        "sizes": {
            "train_queries": len(train_df),
            "val_queries": len(val_df),
            "test_queries": len(test_df),
            "train_passages": len(partitions["train"]),
            "val_passages": len(partitions["val"]),
            "test_passages": len(partitions["test"]),
        },
    }
    if violations:
        raise LeakageError(
            "split leakage audit FAILED — "
            + ", ".join(f"{k}={v}" for k, v in sorted(violations.items()))
        )
    log.info("leakage audit clean: %d checks, 0 violations", len(checks))
    return result
