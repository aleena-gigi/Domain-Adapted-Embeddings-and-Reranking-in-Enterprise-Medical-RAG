#!/usr/bin/env python
"""Phase 1 — Audit and Filter.

Produces the filtered dataset AND the EDA evidence for paper Section 2.
On the critical path: the surviving query count gates Phase 3.

    # full pass (encodes ~232k pairs on the GPU)
    python scripts/01_audit_filter.py

    # skip the encoder pass while iterating on the text filters
    python scripts/01_audit_filter.py --skip-negative-scoring

    # after hand-labeling the validation sample
    python scripts/01_audit_filter.py --validation-labels artifacts/interim/filter_validation_sample_labeled.csv
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
from lyra_capstone.data import eda, filters, load, vocab  # noqa: E402

log = logging.getLogger("phase1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=paths.CONFIGS / "data.yaml")
    ap.add_argument("--skip-negative-scoring", action="store_true",
                    help="skip the bge-m3 pass over (query, negative) pairs")
    ap.add_argument("--validation-labels", type=Path, default=None,
                    help="hand-labeled validation CSV; adds filter precision/recall")
    ap.add_argument("--specialty-sample", type=int, default=None,
                    help="subsample passages for the specialty figure (default: all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(DataConfig, args.config)
    seed = seeds.set_seed(cfg.seed)
    paths.ensure_dirs()
    report: dict = {"seed": seed, "devices": devices.describe()}

    # ---- inputs ----------------------------------------------------------
    df = load.load_medembed_merged(paths.RAW_MEDEMBED)
    corpus_df = load.load_medembed_corpus(paths.RAW_MEDEMBED)
    corpus = dict(zip(corpus_df["id"], corpus_df["text"]))
    log.info("loaded %s queries, %s passages", f"{len(df):,}", f"{len(corpus):,}")
    report["input"] = {"queries": len(df), "passages": len(corpus)}

    # ---- 1.1 anaphoric filter -------------------------------------------
    mesh_path = vocab.download_mesh(paths.INTERIM / "mesh" / vocab.MESH_FILENAME)
    clinical_vocab = vocab.build_clinical_vocab(mesh_path, cfg.anaphoric.min_vocab_term_len)
    report["clinical_vocab"] = {
        "source": str(mesh_path),
        "n_phrases": len(clinical_vocab.phrases),
        "n_unigrams": len(clinical_vocab.unigrams),
        "blocklist_size": len(vocab.GENERIC_MESH_TERMS),
    }

    df = filters.apply_anaphoric_filter(df, clinical_vocab)
    n_ref = int(df["matched_pattern"].notna().sum())
    n_anaph = int(df["is_anaphoric"].sum())
    report["anaphoric"] = {
        "enabled": cfg.anaphoric.enabled,
        "n_with_definite_reference": n_ref,
        "n_flagged_anaphoric": n_anaph,
        "flagged_rate": round(n_anaph / len(df), 6),
        "n_saved_by_clinical_term_clause": n_ref - n_anaph,
        "pattern_breakdown": df["matched_pattern"].value_counts().to_dict(),
        "queries_with_clinical_term_rate": round(float((df["n_clinical_terms"] > 0).mean()), 6),
        # Reported, NOT dropped: outside D5's stated rule. Mostly a measure of
        # MeSH coverage rather than of unanswerability — see audit notes.
        "n_no_mesh_entity_no_reference": int(df["is_generic_no_entity"].sum()),
    }

    validation = filters.validate_anaphoric_filter(
        df, cfg.anaphoric.validation_sample_size, cfg.anaphoric.validation_seed
    )
    validation_path = paths.INTERIM / "filter_validation_sample.csv"
    validation.to_csv(validation_path, index=False)
    log.info("wrote hand-labeling sample -> %s", validation_path)

    agreement = None
    if args.validation_labels:
        agreement = filters.score_filter_agreement(pd.read_csv(args.validation_labels))
        report["anaphoric"]["hand_validation"] = agreement
        log.info("filter vs. hand labels: %s", agreement)
    else:
        report["anaphoric"]["hand_validation"] = None
        log.warning(
            "no hand labels supplied — filter precision/recall UNVALIDATED. "
            "Label %s and re-run with --validation-labels.", validation_path.name
        )

    if cfg.anaphoric.enabled:
        df = df[~df["is_anaphoric"]].copy()
    log.info("after anaphoric filter: %s queries", f"{len(df):,}")

    # ---- 1.3 degenerate positives ---------------------------------------
    if cfg.degenerate_positives.enabled:
        df, degen_stats = filters.filter_degenerate_positives(df, cfg.degenerate_positives.min_tokens)
    else:
        degen_stats = {"enabled": False}
    report["degenerate_positives"] = degen_stats
    log.info("after degenerate-positive filter: %s queries", f"{len(df):,}")

    # ---- 1.2 false negatives --------------------------------------------
    pairs = None
    if cfg.false_negatives.enabled and not args.skip_negative_scoring:
        pairs = filters.score_negative_plausibility(
            df, corpus,
            model_name=cfg.false_negatives.scorer_model,
            batch_size=cfg.false_negatives.encode_batch_size,
        )
        pairs.to_parquet(paths.INTERIM / "negative_scores.parquet")
        percentiles = filters.negative_similarity_percentiles(pairs)
        report["false_negatives"] = {"distribution": percentiles}

        threshold = cfg.false_negatives.sim_threshold
        if threshold is None:
            log.warning(
                "false_negatives.sim_threshold is null — reporting the distribution "
                "only, dropping nothing. Inspect eda_negatives.png, then set the "
                "threshold in configs/data.yaml and re-run (Open Item #4)."
            )
            report["false_negatives"]["applied"] = False
        else:
            df, fn_stats = filters.flag_false_negatives(df, pairs, threshold)
            report["false_negatives"]["applied"] = True
            report["false_negatives"]["stats"] = fn_stats
            log.info("false-negative pruning: %s", fn_stats)
    else:
        report["false_negatives"] = {"skipped": True}

    # ---- outputs ---------------------------------------------------------
    df["n_negatives"] = df["neg_id"].apply(len)
    out_path = paths.INTERIM / "filtered.parquet"
    # clinical_terms is a list column — parquet handles it, but drop the
    # heavier debug columns from the artifact that Phase 2 consumes.
    df.to_parquet(out_path, index=False)

    surviving = len(df)
    report["surviving_queries"] = surviving
    report["surviving_rate_vs_raw"] = round(surviving / report["input"]["queries"], 6)
    report["mean_negatives_per_query"] = round(float(df["n_negatives"].mean()), 3)

    # ---- figures ---------------------------------------------------------
    dpi = cfg.figure_dpi
    figs: dict[str, str] = {}
    figs["eda_lengths"] = str(eda.fig_lengths(df, corpus_df, paths.figure("eda_lengths.png"), dpi))
    # The anaphoric figure describes the full pre-filter population.
    full = filters.apply_anaphoric_filter(load.load_medembed_merged(paths.RAW_MEDEMBED), clinical_vocab)
    figs["eda_anaphoric"] = str(eda.fig_anaphoric(full, paths.figure("eda_anaphoric.png"), dpi, agreement))
    figs["eda_query_types"] = str(eda.fig_query_types(df, paths.figure("eda_query_types.png"), dpi))
    figs["eda_lexical_overlap"] = str(
        eda.fig_lexical_overlap(df, paths.figure("eda_lexical_overlap.png"), dpi)
    )
    if pairs is not None:
        figs["eda_negatives"] = str(
            eda.fig_negatives(pairs, paths.figure("eda_negatives.png"), dpi,
                              cfg.false_negatives.sim_threshold)
        )
    spec_path, spec_stats = eda.fig_specialty(
        corpus_df, clinical_vocab, paths.figure("eda_specialty.png"), dpi, args.specialty_sample
    )
    figs["eda_specialty"] = str(spec_path)
    report["specialty"] = spec_stats
    report["figures"] = figs

    audit_path = paths.INTERIM / "audit_report.json"
    audit_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    manifest.write_manifest(
        paths.INTERIM,
        phase="01_audit_filter",
        config=cfg,
        seed=seed,
        inputs=[paths.RAW_MEDEMBED],
        outputs=[out_path, audit_path, validation_path],
        extra={"surviving_queries": surviving, "figures": figs},
    )

    # ---- acceptance gate -------------------------------------------------
    print("\n" + "=" * 62)
    print(f"Phase 1 complete — SURVIVING QUERIES: {surviving:,} "
          f"({100 * surviving / report['input']['queries']:.1f}% of {report['input']['queries']:,})")
    print(f"  anaphoric dropped   {n_anaph:,}")
    print(f"  degenerate positives dropped {degen_stats.get('n_removed', 0):,}")
    print(f"  mean negatives/query {report['mean_negatives_per_query']}")
    print(f"  audit report        {audit_path}")
    print(f"  hand-label sample   {validation_path}")
    print("=" * 62)

    if surviving < cfg.min_surviving_queries:
        log.error(
            "STOP AND ESCALATE: %s surviving queries is below the %s floor — "
            "the filter is too aggressive (Phase 1 acceptance criteria).",
            f"{surviving:,}", f"{cfg.min_surviving_queries:,}",
        )
        return 2
    if agreement is None:
        print("\nNEXT: label artifacts/interim/filter_validation_sample.csv "
              "(human_label: 1 = anaphoric, 0 = answerable), then re-run with "
              "--validation-labels to fill in filter precision/recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
