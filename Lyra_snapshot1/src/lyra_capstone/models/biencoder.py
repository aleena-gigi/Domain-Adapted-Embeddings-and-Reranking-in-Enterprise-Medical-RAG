"""Phase 3 — Experiment A: domain-adapt bge-m3 (spec detail §3, spec v2 §3).

`CachedMultipleNegativesRankingLoss` is not optional here. In-batch negatives
are the dominant training signal for contrastive retrieval, and GradCache is
what buys an effective batch of 256 on one 48 GB card at XLM-R-large scale
(~568M params). The mined hard negatives ride along on top of that.

Two things in this module are choices rather than transcriptions of the spec,
both recorded in docs/decisions.md:

  * the in-training validation index is the **val partition** (2,759 passages),
    not the full 27,590-passage corpus. It is a model-selection signal, and
    re-encoding the whole corpus at every eval step would cost more than the
    training itself. The winning config is re-scored against the full index
    once, at the end, so the gap between the two is measured rather than
    assumed;
  * rows with fewer negatives than the configured `n_hard_negatives` are
    dropped, not padded. Duplicating a negative to fill a column would put
    fabricated data in the loss.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BiEncoderConfig:
    base_model: str = "BAAI/bge-m3"
    max_seq_length: int = 320        # passages cap at 993 chars; 8192 wastes memory
    mini_batch_size: int = 32        # physical, per GradCache step
    effective_batch_size: int = 256  # what the loss actually sees
    learning_rate: float = 2e-5
    epochs: int = 2
    warmup_ratio: float = 0.1
    n_hard_negatives: int = 8
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    # --- beyond the spec's dataclass, needed to actually run -------------
    evals_per_epoch: int = 4
    # Guards divergence rather than trimming epochs: the sweep's `epochs` axis
    # is only meaningful if a run is allowed to finish its schedule.
    early_stopping_patience: int = 4
    eval_batch_size: int = 128
    scale: float = 20.0              # 1/temperature for the contrastive loss
    save_total_limit: int = 1
    # Smoke-testing only: caps the run at N optimizer steps. -1 is "honour
    # `epochs`", which is what every recorded run uses.
    max_steps: int = -1
    # A 568M checkpoint is ~2.3 GB and the best one is re-saved under model/,
    # so the trainer's checkpoint tree is deleted once training finishes.
    keep_checkpoints: bool = False

    @property
    def run_id(self) -> str:
        return (f"bi_lr{self.learning_rate:g}_ep{self.epochs}"
                f"_neg{self.n_hard_negatives}_s{self.seed}")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_training_dataset(path: Path, n_negatives: int):
    """train.jsonl -> (anchor, positive, negative_1..n) columns.

    The column layout is positional for MultipleNegativesRankingLoss, so the
    width has to be fixed. Rows carrying fewer than `n_negatives` are dropped
    and counted — see the module docstring.
    """
    from datasets import Dataset

    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    usable = [r for r in rows if len(r["neg"]) >= n_negatives]
    if len(usable) < len(rows):
        log.info(
            "dropped %d/%d rows with fewer than %d negatives",
            len(rows) - len(usable), len(rows), n_negatives,
        )

    data = {"anchor": [r["query"] for r in usable],
            "positive": [r["pos"] for r in usable]}
    for i in range(n_negatives):
        data[f"negative_{i + 1}"] = [r["neg"][i] for r in usable]
    return Dataset.from_dict(data), {"n_rows": len(usable), "n_dropped": len(rows) - len(usable)}


def build_ir_evaluator(
    queries_path: Path,
    corpus_path: Path,
    name: str,
    restrict_to_split: str | None = None,
    batch_size: int = 128,
    ndcg_at_k: tuple[int, ...] = (10,),
):
    """InformationRetrievalEvaluator over a jsonl split.

    `restrict_to_split` limits the index to one partition — used for the
    in-training val signal. Pass None to index the full corpus, which is what
    the results table uses.
    """
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    qrows = [json.loads(line) for line in Path(queries_path).read_text().splitlines()]
    crows = [json.loads(line) for line in Path(corpus_path).read_text().splitlines()]
    if restrict_to_split:
        crows = [c for c in crows if c["split"] == restrict_to_split]

    corpus = {c["id"]: c["text"] for c in crows}
    missing = [q for q in qrows if q["pos_id"] not in corpus]
    if missing:
        raise ValueError(
            f"{len(missing)} {name} queries have a pos_id outside the index "
            f"(restrict_to_split={restrict_to_split!r})"
        )

    log.info("%s evaluator: %s queries over %s passages",
             name, f"{len(qrows):,}", f"{len(corpus):,}")
    return InformationRetrievalEvaluator(
        queries={q["query_id"]: q["query"] for q in qrows},
        corpus=corpus,
        relevant_docs={q["query_id"]: {q["pos_id"]} for q in qrows},
        name=name,
        batch_size=batch_size,
        ndcg_at_k=list(ndcg_at_k),
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        mrr_at_k=[10],
        show_progress_bar=False,
        write_csv=False,
        main_score_function="cosine",
    )


def ndcg_key(metrics: dict, name: str) -> str:
    """Locate the nDCG@10 key in an evaluator's output.

    sentence-transformers has renamed these between majors; probing beats
    hardcoding a string that silently stops matching and leaves
    `metric_for_best_model` pointing at nothing.
    """
    for k in metrics:
        if "ndcg@10" in k.lower() and "cosine" in k.lower():
            return k
    for k in metrics:
        if "ndcg@10" in k.lower():
            return k
    raise KeyError(f"no nDCG@10 in {name} evaluator output: {sorted(metrics)}")


def assert_embedding_dim(model, expected: int = 1024) -> int:
    """Probe a real embedding. Never trust the config field (spec v2 §3)."""
    dim = int(model.encode(["dimension probe"], convert_to_numpy=True).shape[-1])
    if dim != expected:
        raise ValueError(
            f"embedding dim {dim} != {expected}; the Milvus collection contract "
            "assumes 1024-dim COSINE"
        )
    return dim


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train_biencoder(
    cfg: BiEncoderConfig,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    corpus_path: Path | None = None,
    figure_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Fine-tune bge-m3 with CachedMultipleNegativesRankingLoss.

    Evaluates nDCG@10 on val every few hundredths of an epoch, early-stops on
    it, and saves the **best** checkpoint rather than the last. Writes
    `metrics.json` beside the model and returns the model directory.
    """
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
    )
    from sentence_transformers.training_args import BatchSamplers
    from transformers import EarlyStoppingCallback

    from .. import devices, paths, seeds

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_path or (paths.PROCESSED / "corpus.jsonl")
    seeds.set_seed(cfg.seed)
    device = devices.train_device()

    train_ds, data_stats = load_training_dataset(train_path, cfg.n_hard_negatives)
    steps_per_epoch = max(1, len(train_ds) // cfg.effective_batch_size)
    eval_steps = max(1, steps_per_epoch // cfg.evals_per_epoch)

    model = SentenceTransformer(
        cfg.base_model, device=device, model_kwargs=devices.sdpa_model_kwargs(device)
    )
    model.max_seq_length = cfg.max_seq_length
    dim = assert_embedding_dim(model)

    evaluator = build_ir_evaluator(
        val_path, corpus_path, name="val",
        restrict_to_split="val", batch_size=cfg.eval_batch_size,
    )
    baseline = evaluator(model)
    metric = ndcg_key(baseline, "val")
    log.info("stock %s = %.4f (pre-training baseline)", metric, baseline[metric])

    loss = CachedMultipleNegativesRankingLoss(
        model, scale=cfg.scale, mini_batch_size=cfg.mini_batch_size
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.effective_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        # transformers v5 folded `warmup_ratio` into a float `warmup_steps`.
        warmup_steps=float(cfg.warmup_ratio),
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # MNRL treats every other row in the batch as a negative, so a repeated
        # passage inside one batch becomes a false negative in the loss itself.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{metric}",
        greater_is_better=True,
        logging_steps=1,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        disable_tqdm=True,  # sweeps run detached; logging_steps=1 is the record
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    trainer.train()
    wall = time.time() - started
    peak_vram = devices.peak_vram_gb()

    model.save(str(out_dir / "model"))
    final = evaluator(model)

    metrics = {
        "run_id": cfg.run_id,
        "config": asdict(cfg),
        "device": device,
        "embedding_dim": dim,
        "data": data_stats,
        "steps_per_epoch": steps_per_epoch,
        "eval_steps": eval_steps,
        "train_seconds": round(wall, 1),
        "peak_vram_gb": peak_vram,
        "metric": metric,
        "stock_baseline": {k: float(v) for k, v in baseline.items() if isinstance(v, (int, float))},
        "best": {k: float(v) for k, v in final.items() if isinstance(v, (int, float))},
        "val_ndcg@10_stock": float(baseline[metric]),
        "val_ndcg@10_best": float(final[metric]),
        "val_index": "val partition only (model selection); see docs/decisions.md P3a",
        "log_history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    if figure_path is not None:
        plot_training_curve(trainer.state.log_history, metric, figure_path, cfg.run_id, dpi)

    log.info(
        "%s: val nDCG@10 %.4f -> %.4f in %.1f min, peak VRAM %.1f GB",
        cfg.run_id, baseline[metric], final[metric], wall / 60, peak_vram,
    )

    del trainer, loss, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not cfg.keep_checkpoints:
        import shutil

        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return out_dir


def plot_training_curve(log_history: list[dict], metric: str, out_path: Path,
                        title: str, dpi: int = 300) -> Path:
    """Loss against val nDCG@10 — paper Section 5 asks for training curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..data.eda import PALETTE

    loss = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    key = f"eval_{metric}"
    evals = [(e["step"], e[key]) for e in log_history if key in e]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if loss:
        ax.plot(*zip(*loss), color=PALETTE["primary"], lw=1.2, label="train loss")
    ax.set_xlabel("step")
    ax.set_ylabel("contrastive loss", color=PALETTE["primary"])
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    if evals:
        ax2 = ax.twinx()
        ax2.plot(*zip(*evals), color=PALETTE["secondary"], lw=1.6,
                 marker="o", ms=3.5, label="val nDCG@10")
        best_step, best_val = max(evals, key=lambda t: t[1])
        ax2.axvline(best_step, color=PALETTE["muted"], ls="--", lw=0.9)
        ax2.annotate(f"best {best_val:.4f}", (best_step, best_val),
                     textcoords="offset points", xytext=(6, -12), fontsize=8,
                     color=PALETTE["secondary"])
        ax2.set_ylabel("val nDCG@10", color=PALETTE["secondary"])
        ax2.spines[["top"]].set_visible(False)

    ax.set_title(f"Experiment A — {title}", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

# The spec's suggested grid (§3). 2 x 3 x 2 = 12 runs.
DEFAULT_GRID = {
    "learning_rate": [1e-5, 2e-5],
    "epochs": [1, 2, 3],
    "n_hard_negatives": [4, 8],
}


def run_sweep(
    base_cfg: BiEncoderConfig,
    grid: dict[str, list],
    out_dir: Path,
    train_path: Path | None = None,
    val_path: Path | None = None,
    corpus_path: Path | None = None,
    figures_dir: Path | None = None,
    dpi: int = 300,
    resume: bool = True,
) -> pd.DataFrame:
    """Grid over learning_rate x epochs x n_hard_negatives.

    Every point is a separate run with its own LR schedule. Reading epoch
    counts off one long run's checkpoints would be cheaper, but the schedule
    warms up and decays over the *configured* horizon, so a checkpoint at epoch
    1 of a 3-epoch run is not the model a 1-epoch run produces. The sweep is
    the paper's comparative-optimization content; it should not compare things
    that were not actually trained.

    Returns a dataframe of (config, val nDCG@10) for the results section.
    """
    from .. import paths

    out_dir = Path(out_dir)
    train_path = train_path or (paths.PROCESSED / "train.jsonl")
    val_path = val_path or (paths.PROCESSED / "val.jsonl")
    corpus_path = corpus_path or (paths.PROCESSED / "corpus.jsonl")
    figures_dir = figures_dir or paths.FIGURES

    keys = list(grid)
    points = [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]
    log.info("sweep: %d configurations", len(points))

    rows = []
    for i, point in enumerate(points, 1):
        cfg = replace(base_cfg, **point)
        run_out = out_dir / cfg.run_id
        metrics_path = run_out / "metrics.json"

        if resume and metrics_path.exists():
            log.info("[%d/%d] %s already trained — reusing", i, len(points), cfg.run_id)
            metrics = json.loads(metrics_path.read_text())
        else:
            log.info("[%d/%d] training %s", i, len(points), cfg.run_id)
            train_biencoder(
                cfg, train_path, val_path, run_out,
                corpus_path=corpus_path,
                figure_path=figures_dir / f"train_curve_{cfg.run_id}.png",
                dpi=dpi,
            )
            metrics = json.loads(metrics_path.read_text())

        rows.append({
            "run_id": cfg.run_id,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "n_hard_negatives": cfg.n_hard_negatives,
            "seed": cfg.seed,
            "val_ndcg@10": metrics["val_ndcg@10_best"],
            "gain_vs_stock": round(
                metrics["val_ndcg@10_best"] - metrics["val_ndcg@10_stock"], 4
            ),
            "train_minutes": round(metrics["train_seconds"] / 60, 1),
            "peak_vram_gb": metrics["peak_vram_gb"],
            "train_rows": metrics["data"]["n_rows"],
            "model_dir": str(run_out / "model"),
        })

    return pd.DataFrame(rows).sort_values("val_ndcg@10", ascending=False).reset_index(drop=True)
