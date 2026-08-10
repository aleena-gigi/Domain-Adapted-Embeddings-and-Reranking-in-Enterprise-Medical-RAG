"""
Experiment A — fine-tune BAAI/bge-m3 as a bi-encoder retriever on the cleaned
MedEmbed clinical triplets (AAI-590 capstone, Domain-Adapted Embeddings and
Reranking in Enterprise Medical RAG).

Reads the cleaned, leak-checked train/val/test splits written by
02_data_cleaning-2.ipynb (data/processed/{train,val,test}.jsonl and
corpus_clean.jsonl) and fine-tunes bge-m3 with a multiple-negatives-ranking
contrastive loss. Runs a small hyperparameter sweep over learning rate and
epochs, selecting the best checkpoint by validation nDCG@10, and writes the
final model plus a sweep log that the Results section reads from.

This script does not evaluate the fine-tuned model against the baseline
configurations (BM25, stock bge-m3, MedEmbed-large) -- that comparison
belongs to the evaluation harness (04_evaluate.py), so training results here
are reported only as validation nDCG@10 during the sweep, not as final
comparative numbers.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

DATA_DIR = Path("../data/processed")
MODEL_DIR = Path("../models")
LOG_DIR = Path("../logs")
BASE_MODEL = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 320


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_train_examples(train_records: list[dict], negatives_per_query: int) -> list[InputExample]:
    """One InputExample per (query, positive, negative) triplet, capped at
    negatives_per_query negatives per query for this run of the sweep."""
    examples = []
    for rec in train_records:
        negs = rec["negs"][:negatives_per_query]
        for neg in negs:
            examples.append(InputExample(texts=[rec["query"], rec["pos"], neg]))
    return examples


def build_ir_evaluator(records: list[dict], corpus: dict[str, str], name: str) -> InformationRetrievalEvaluator:
    """Build a sentence-transformers IR evaluator: each query's relevant docs
    are its accepted positive id(s); the corpus is every passage assigned to
    this split plus the shared distractor passages (per the cleaning
    notebook's leak-safe per-split retrieval corpus)."""
    queries = {rec["query_id"]: rec["query"] for rec in records}
    relevant_docs = {
        rec["query_id"]: set(rec["alt_pos_ids"]) | {rec["pos_id"]} for rec in records
    }
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
        ndcg_at_k=[10],
        precision_recall_at_k=[5, 10],
        mrr_at_k=[10],
        batch_size=32,
    )


def train_one_config(
    train_examples: list[InputExample],
    val_evaluator: InformationRetrievalEvaluator,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    warmup_ratio: float,
    run_name: str,
) -> dict:
    model = SentenceTransformer(BASE_MODEL, device=device())
    model.max_seq_length = MAX_SEQ_LENGTH

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    n_steps = len(train_dataloader) * epochs
    warmup_steps = int(n_steps * warmup_ratio)

    output_path = str(MODEL_DIR / run_name)
    start = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        evaluator=val_evaluator,
        evaluation_steps=max(1, len(train_dataloader) // 2),
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    val_ndcg = val_evaluator(model, output_path=str(LOG_DIR))
    return {
        "run_name": run_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "val_ndcg_at_10": float(val_ndcg) if isinstance(val_ndcg, (int, float)) else val_ndcg,
        "train_seconds": elapsed,
        "n_train_examples": len(train_examples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--sweep",
        type=str,
        default="lr=1e-5,epochs=1;lr=2e-5,epochs=1;lr=2e-5,epochs=2",
        help="semicolon-separated lr=,epochs= pairs",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device()}")

    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    val_records = load_jsonl(DATA_DIR / "val.jsonl")
    corpus_records = load_jsonl(DATA_DIR / "corpus_clean.jsonl")

    # Per-split retrieval corpus: this split's positives plus every passage marked "shared".
    val_corpus = {
        r["id"]: r["text"] for r in corpus_records if r["split"] in ("val", "shared")
    }
    print(f"Train queries: {len(train_records):,}")
    print(f"Val queries: {len(val_records):,}, val corpus: {len(val_corpus):,} passages")

    train_examples = build_train_examples(train_records, args.negatives_per_query)
    print(f"Train examples (query, pos, neg) triplets: {len(train_examples):,}")

    val_evaluator = build_ir_evaluator(val_records, val_corpus, name="val")

    sweep_configs = []
    for spec in args.sweep.split(";"):
        parts = dict(kv.split("=") for kv in spec.split(","))
        sweep_configs.append({"lr": float(parts["lr"]), "epochs": int(parts["epochs"])})

    results = []
    for i, cfg in enumerate(sweep_configs):
        run_name = f"bge_m3_ft_lr{cfg['lr']}_ep{cfg['epochs']}"
        print(f"\n=== Sweep {i + 1}/{len(sweep_configs)}: {run_name} ===")
        result = train_one_config(
            train_examples=train_examples,
            val_evaluator=val_evaluator,
            learning_rate=cfg["lr"],
            epochs=cfg["epochs"],
            batch_size=args.batch_size,
            warmup_ratio=0.1,
            run_name=run_name,
        )
        print(json.dumps(result, indent=2))
        results.append(result)

        with open(LOG_DIR / "sweep_results.json", "w") as fh:
            json.dump(results, fh, indent=2)

    best = max(results, key=lambda r: r["val_ndcg_at_10"])
    print(f"\nBest config by val nDCG@10: {best['run_name']} ({best['val_ndcg_at_10']:.4f})")

    best_model_path = MODEL_DIR / best["run_name"]
    final_path = MODEL_DIR / "bge_m3_finetuned_best"
    if final_path.exists():
        import shutil

        shutil.rmtree(final_path)
    import shutil

    shutil.copytree(best_model_path, final_path)
    print(f"Copied best model to {final_path}")


if __name__ == "__main__":
    main()
