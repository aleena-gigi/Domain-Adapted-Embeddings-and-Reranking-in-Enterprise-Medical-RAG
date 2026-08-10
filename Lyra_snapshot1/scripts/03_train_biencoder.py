#!/usr/bin/env python
"""Phase 3 — Experiment A: bi-encoder training and hyperparameter sweep.

Domain-adapts bge-m3 with CachedMultipleNegativesRankingLoss, sweeps
learning rate x epochs x hard-negatives, and records the comparative table the
paper's optimization section needs.

    # the full 12-point sweep (resumes anything already trained)
    python scripts/03_train_biencoder.py

    # one run at the config's own hyperparameters
    python scripts/03_train_biencoder.py --single

    # plumbing check: a handful of steps on a slice of the data
    python scripts/03_train_biencoder.py --single --max-steps 3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyra_capstone import devices, manifest, paths, seeds  # noqa: E402
from lyra_capstone.config import TrainAConfig, load_config  # noqa: E402
from lyra_capstone.models import biencoder  # noqa: E402

log = logging.getLogger("phase3")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=paths.CONFIGS / "biencoder.yaml")
    ap.add_argument("--single", action="store_true",
                    help="train one model at the config's hyperparameters instead of sweeping")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (smoke-testing only; not for recorded runs)")
    ap.add_argument("--no-resume", action="store_true",
                    help="retrain sweep points that already have metrics.json")
    ap.add_argument("--skip-full-index-check", action="store_true",
                    help="skip re-scoring the winner against the full 27,590-passage index")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(TrainAConfig, args.config)
    seed = seeds.set_seed(cfg.seed)
    paths.ensure_dirs()

    base = replace(cfg.model, seed=cfg.seed)
    if args.max_steps is not None:
        base = replace(base, max_steps=args.max_steps)
        log.warning("--max-steps %d: this is a smoke test, not a recorded run", args.max_steps)

    train_path = paths.PROCESSED / "train.jsonl"
    val_path = paths.PROCESSED / "val.jsonl"
    corpus_path = paths.PROCESSED / "corpus.jsonl"
    for p in (train_path, val_path, corpus_path):
        if not p.exists():
            log.error("missing %s — run scripts/02_build_splits.py first", p)
            return 1

    grid = ({k: [getattr(base, k)] for k in ("learning_rate", "epochs", "n_hard_negatives")}
            if args.single else asdict(cfg.sweep))

    table = biencoder.run_sweep(
        base, grid, paths.BIENCODER_MODELS,
        train_path=train_path, val_path=val_path, corpus_path=corpus_path,
        figures_dir=paths.FIGURES, dpi=cfg.figure_dpi,
        resume=not args.no_resume,
    )

    # ---- sweep artifacts -------------------------------------------------
    sweep_path = paths.RUNS / "sweep_biencoder.csv"
    table.to_csv(sweep_path, index=False)
    fig_path = plot_sweep(table, paths.figure("sweep_biencoder.png"), cfg.figure_dpi)

    best = table.iloc[0]
    best_run = best["run_id"]
    best_dir = paths.BIENCODER_MODELS / best_run
    metrics = json.loads((best_dir / "metrics.json").read_text())

    # Mirror the winner's metrics into the run registry (spec detail §3 outputs).
    run_dir = paths.run_dir(best_run)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_dir / "metrics.json", run_dir / "metrics.json")

    # ---- anchor the selection signal against the real index --------------
    # In-training val nDCG@10 is measured over the 2,759-passage val partition,
    # which is not the index the results table uses. Score the winner once over
    # the full corpus so the gap between the two is a number, not a hope.
    full_index = None
    if not args.skip_full_index_check:
        from sentence_transformers import SentenceTransformer

        infer_device = devices.infer_device()
        model = SentenceTransformer(
            str(best_dir / "model"), device=infer_device,
            model_kwargs=devices.sdpa_model_kwargs(infer_device),
        )
        model.max_seq_length = base.max_seq_length
        evaluator = biencoder.build_ir_evaluator(
            val_path, corpus_path, name="val_full", restrict_to_split=None,
            batch_size=base.eval_batch_size,
        )
        scores = evaluator(model)
        key = biencoder.ndcg_key(scores, "val_full")
        full_index = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        log.info("winner on the FULL 27,590-passage index: %s = %.4f", key, scores[key])
        del model

    pointer = {
        "run_id": best_run,
        "model_dir": str(best_dir / "model"),
        "selected_by": "val nDCG@10 on the val-partition index",
        "val_ndcg@10_val_partition_index": float(best["val_ndcg@10"]),
        "val_ndcg@10_full_index": (
            None if full_index is None else full_index[biencoder.ndcg_key(full_index, "val_full")]
        ),
        "stock_baseline_val_partition_index": metrics["val_ndcg@10_stock"],
        "config": metrics["config"],
    }
    (paths.BIENCODER_MODELS / "best.json").write_text(json.dumps(pointer, indent=2) + "\n")

    manifest.write_manifest(
        paths.BIENCODER_MODELS,
        phase="03_train_biencoder",
        config=cfg,
        seed=seed,
        inputs=[train_path, val_path, corpus_path],
        outputs=[sweep_path, fig_path, paths.BIENCODER_MODELS / "best.json"],
        extra={
            "n_runs": int(len(table)),
            "best": pointer,
            "full_index_scores": full_index,
            "peak_vram_gb": float(table["peak_vram_gb"].max()),
            "total_train_minutes": float(table["train_minutes"].sum()),
        },
    )

    print("\n" + "=" * 72)
    print(f"Phase 3 complete — {len(table)} runs, "
          f"{table['train_minutes'].sum():.0f} min total, "
          f"peak VRAM {table['peak_vram_gb'].max():.1f} GB")
    print(table.drop(columns=["model_dir", "seed"]).to_string(index=False))
    print(f"\n  best        {best_run}")
    print(f"  val nDCG@10 {metrics['val_ndcg@10_stock']:.4f} (stock) -> "
          f"{best['val_ndcg@10']:.4f}  [val-partition index]")
    if full_index is not None:
        print(f"  same model on the full 27,590-passage index: "
              f"{pointer['val_ndcg@10_full_index']:.4f}")
    print(f"  sweep table {sweep_path}")
    print(f"  pointer     {paths.BIENCODER_MODELS / 'best.json'}")
    print("=" * 72)
    return 0


def plot_sweep(table, out_path: Path, dpi: int) -> Path:
    """val nDCG@10 across the grid — the paper's optimization figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lyra_capstone.data.eda import PALETTE

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    styles = [
        (PALETTE["primary"], "o", "-"), (PALETTE["secondary"], "s", "-"),
        (PALETTE["accent"], "^", "--"), (PALETTE["warn"], "D", "--"),
    ]
    groups = sorted(table.groupby(["learning_rate", "n_hard_negatives"]).groups)
    for (lr, neg), (color, marker, ls) in zip(groups, styles):
        sub = table[(table["learning_rate"] == lr) & (table["n_hard_negatives"] == neg)]
        sub = sub.sort_values("epochs")
        ax.plot(sub["epochs"], sub["val_ndcg@10"], color=color, marker=marker,
                ls=ls, lw=1.5, ms=5, label=f"lr={lr:g}, neg={neg}")

    ax.set_xlabel("epochs")
    ax.set_ylabel("val nDCG@10 (val-partition index)")
    ax.set_xticks(sorted(table["epochs"].unique()))
    ax.set_title("Experiment A — hyperparameter sweep", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
