#!/usr/bin/env python
"""Phase 2 — Splits and Dataset Build.

Document-level 80/10/10 partition of the corpus, queries inheriting their
positive's partition, training negatives sanitized and re-mined from the train
partition only. Ends in a hard leakage audit that raises rather than warns.

    # first pass: report the near-duplicate distribution, flag nothing
    python scripts/02_build_splits.py

    # after setting splits.semantic_dup_threshold in configs/data.yaml
    python scripts/02_build_splits.py

    # skip the GPU passes while iterating on partition logic
    python scripts/02_build_splits.py --skip-dup-audit --skip-mining
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyra_capstone import devices, manifest, paths, seeds  # noqa: E402
from lyra_capstone.config import DataConfig, load_config  # noqa: E402
from lyra_capstone.data import load, splits  # noqa: E402

log = logging.getLogger("phase2")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=paths.CONFIGS / "data.yaml")
    ap.add_argument("--skip-dup-audit", action="store_true",
                    help="skip the encoder pass for cross-split near-duplicates")
    ap.add_argument("--skip-mining", action="store_true",
                    help="strip leaking negatives but do not re-mine replacements")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(DataConfig, args.config)
    seed = seeds.set_seed(cfg.seed)
    paths.ensure_dirs()
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": seed, "devices": devices.describe()}

    # ---- inputs ----------------------------------------------------------
    filtered_path = paths.INTERIM / "filtered.parquet"
    if not filtered_path.exists():
        log.error("missing %s — run scripts/01_audit_filter.py first", filtered_path)
        return 1
    df = pd.read_parquet(filtered_path)
    corpus_df = load.load_medembed_corpus(paths.RAW_MEDEMBED)
    corpus = dict(zip(corpus_df["id"], corpus_df["text"]))
    log.info("loaded %s filtered queries, %s passages", f"{len(df):,}", f"{len(corpus):,}")
    report["input"] = {"queries": len(df), "passages": len(corpus)}

    # ---- 2.1 partition ---------------------------------------------------
    partitions = splits.partition_corpus(
        corpus_df["id"].tolist(), tuple(cfg.split_ratios), seed
    )
    df = splits.assign_queries_to_splits(df, partitions)

    # ---- 2.2 cross-split duplicate queries -------------------------------
    if args.skip_dup_audit:
        semantic_pairs, dup_stats = pd.DataFrame(), {"skipped": True}
    else:
        semantic_pairs, dup_stats = splits.find_semantic_duplicate_queries(
            df,
            threshold=cfg.splits.semantic_dup_threshold,
            model_name=cfg.splits.scorer_model,
            batch_size=cfg.splits.encode_batch_size,
            chunk=cfg.splits.dup_chunk_size,
        )
        # Kept whole, not just the flagged tail: the threshold has to be
        # defensible from the bands below it as well as above.
        pairs_path = paths.INTERIM / "near_duplicate_pairs.parquet"
        semantic_pairs.to_parquet(pairs_path, index=False)
        dup_stats["pairs_artifact"] = str(pairs_path)
    report["near_duplicates"] = dup_stats

    df, resolution = splits.resolve_cross_split_duplicates(df, semantic_pairs)
    report["duplicate_resolution"] = resolution

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # ---- 2.3 training-negative sanitation --------------------------------
    # null in config means "inherit Phase 1's false-negative threshold" —
    # top-k mining aims straight at the region Phase 1 pruned.
    cap = cfg.splits.mine_max_similarity
    if cap is None:
        cap = cfg.false_negatives.sim_threshold

    if args.skip_mining:
        train_df["neg_id"] = [
            [n for n in splits._as_list(negs) if n in partitions["train"] and n != pid]
            for negs, pid in zip(train_df["neg_id"], train_df["pos_id"])
        ]
        train_df["neg"] = [[corpus[i] for i in ids] for ids in train_df["neg_id"]]
        mining_stats = {"skipped": True}
    else:
        train_df, mining_stats = splits.sanitize_training_negatives(
            train_df,
            partitions,
            corpus,
            n_negatives=cfg.n_negatives_per_query,
            model_name=cfg.splits.scorer_model,
            max_similarity=cap,
            batch_size=cfg.splits.encode_batch_size,
            chunk=cfg.splits.mine_chunk_size,
        )
    report["training_negatives"] = mining_stats

    # ---- 2.4 leakage audit — raises --------------------------------------
    # The semantic residual is 0 by construction once the train twins are
    # dropped; the check is here so a bug in the resolution step surfaces as a
    # leakage failure rather than as a quietly contaminated training set.
    residual = 0
    if not semantic_pairs.empty and "flagged" in semantic_pairs:
        alive = set(df["query_id"])
        flagged = semantic_pairs[semantic_pairs["flagged"]]
        residual = int(sum(
            r.query_id in alive and r.other_query_id in alive
            for r in flagged.itertuples(index=False)
        ))
    audit = splits.audit_leakage(
        train_df, val_df, test_df, partitions,
        extra_violations={"semantic_near_duplicate_queries_across_splits": residual},
    )
    report["leakage_audit"] = audit

    # ---- outputs ---------------------------------------------------------
    out_paths = []
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_paths.append(
            splits.write_jsonl(paths.PROCESSED / f"{name}.jsonl", splits.query_records(frame))
        )
    corpus_path = splits.write_jsonl(
        paths.PROCESSED / "corpus.jsonl", splits.corpus_records(corpus_df, partitions)
    )
    out_paths.append(corpus_path)

    report["outputs"] = {p.name: p.stat().st_size for p in out_paths}
    split_manifest = paths.PROCESSED / "split_manifest.json"
    split_manifest.write_text(json.dumps(report, indent=2, default=str) + "\n")

    manifest.write_manifest(
        paths.PROCESSED,
        phase="02_build_splits",
        config=cfg,
        seed=seed,
        inputs=[filtered_path, paths.RAW_MEDEMBED],
        outputs=[*out_paths, split_manifest],
        extra={"sizes": audit["sizes"], "n_violations": audit["n_violations"]},
    )

    # ---- summary ---------------------------------------------------------
    s = audit["sizes"]
    print("\n" + "=" * 62)
    print("Phase 2 complete — leakage audit: "
          f"{len(audit['checks'])} checks, {audit['n_violations']} violations")
    print(f"  passages  train {s['train_passages']:,}  "
          f"val {s['val_passages']:,}  test {s['test_passages']:,}")
    print(f"  queries   train {s['train_queries']:,}  "
          f"val {s['val_queries']:,}  test {s['test_queries']:,}")
    if not mining_stats.get("skipped"):
        print(f"  train negatives  {mining_stats['negatives_dropped_out_of_partition']:,} "
              f"dropped out-of-partition, {mining_stats['negatives_mined']:,} re-mined, "
              f"mean {mining_stats['mean_negatives_after']}/query")
    print(f"  split manifest   {split_manifest}")
    print("=" * 62)

    if cfg.splits.semantic_dup_threshold is None and not args.skip_dup_audit:
        print("\nNEXT: near-duplicate threshold is unset. Inspect "
              "near_duplicates.top_pairs in split_manifest.json, set "
              "splits.semantic_dup_threshold in configs/data.yaml, re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
