"""Phase 1 EDA figures (spec detail §1.4) — paper Section 2.

Six figures, all at publication quality: 300 dpi, light background, readable at
single-column width. Notebooks import from here and never define analysis logic
inline (§0.1).
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .filters import classify_query_type, count_tokens, lexical_overlap  # noqa: E402
from .vocab import DISEASE_SUBCATEGORIES, ClinicalVocab  # noqa: E402

log = logging.getLogger(__name__)

# Colorblind-safe, prints legibly in greyscale.
PALETTE = {
    "primary": "#3B6EA8",
    "secondary": "#C8674B",
    "accent": "#5E8C61",
    "muted": "#8A8F98",
    "warn": "#B5563F",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def _save(fig, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------


def fig_lengths(df: pd.DataFrame, corpus: pd.DataFrame, out: Path, dpi: int = 300) -> Path:
    """Query and passage length distributions."""
    _style()
    q_chars = df["query"].str.len()
    p_chars = corpus["text"].str.len()
    q_tok = df["query"].map(count_tokens)
    p_tok = corpus["text"].map(count_tokens)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, data, title, color in (
        (axes[0, 0], q_chars, "Query length (characters)", PALETTE["primary"]),
        (axes[0, 1], p_chars, "Passage length (characters)", PALETTE["secondary"]),
        (axes[1, 0], q_tok, "Query length (tokens)", PALETTE["primary"]),
        (axes[1, 1], p_tok, "Passage length (tokens)", PALETTE["secondary"]),
    ):
        ax.hist(data, bins=60, color=color, edgecolor="none")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.axvline(
            data.median(), color="black", lw=1, ls="--", label=f"median {data.median():.0f}"
        )
        ax.legend(loc="upper right", fontsize=8)

    # The max_seq_length=320 choice in Experiment A rests on this tail.
    axes[1, 1].axvline(320, color=PALETTE["warn"], lw=1.2, ls=":", label="max_seq_length 320")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Query and passage length distributions", y=1.0, fontsize=12)
    return _save(fig, out, dpi)


def fig_anaphoric(df: pd.DataFrame, out: Path, dpi: int = 300, agreement: dict | None = None) -> Path:
    """Anaphoric rate, pattern breakdown, and (once labeled) filter precision/recall."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    n = len(df)
    n_ref = int(df["matched_pattern"].notna().sum())
    n_anaph = int(df["is_anaphoric"].sum())
    n_saved = n_ref - n_anaph

    ax = axes[0]
    bars = ["No patient\nreference", "Reference,\nsaved by\nclinical term", "Anaphoric\n(dropped)"]
    vals = [n - n_ref, n_saved, n_anaph]
    colors = [PALETTE["muted"], PALETTE["accent"], PALETTE["warn"]]
    b = ax.bar(bars, vals, color=colors)
    ax.bar_label(b, labels=[f"{v:,}\n{100*v/n:.1f}%" for v in vals], fontsize=8, padding=2)
    ax.set_title("Anaphoric filter outcome")
    ax.set_ylabel("queries")
    ax.set_ylim(0, max(vals) * 1.22)

    ax = axes[1]
    counts = df["matched_pattern"].value_counts()
    ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE["primary"])
    ax.set_title("Definite-reference pattern hit")
    ax.set_xlabel("queries")

    ax = axes[2]
    if agreement:
        keys = ["precision", "recall", "f1", "accuracy"]
        vals = [agreement.get(k, 0.0) for k in keys]
        b = ax.bar(keys, vals, color=PALETTE["accent"])
        ax.bar_label(b, fmt="%.2f", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Filter vs. hand labels (n={agreement.get('n_labeled', 0)})")
    else:
        ax.text(
            0.5, 0.5,
            "Hand validation pending\n\nLabel "
            "filter_validation_sample.csv,\nthen re-run with --validation-labels",
            ha="center", va="center", fontsize=9, color=PALETTE["muted"],
        )
        ax.set_axis_off()
        ax.set_title("Filter precision / recall")

    fig.suptitle("Anaphoric query filter (D5)", y=1.02, fontsize=12)
    return _save(fig, out, dpi)


def fig_negatives(pairs: pd.DataFrame, out: Path, dpi: int = 300, threshold: float | None = None) -> Path:
    """Negative similarity distribution + reuse frequency (Issue 2)."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    sims = pairs["neg_similarity"].to_numpy()
    ax = axes[0]
    ax.hist(sims, bins=80, color=PALETTE["primary"], edgecolor="none")
    for q, ls in ((95, ":"), (99, "--")):
        v = float(np.percentile(sims, q))
        ax.axvline(v, color="black", lw=1, ls=ls, label=f"p{q} = {v:.3f}")
    if threshold is not None:
        ax.axvline(threshold, color=PALETTE["warn"], lw=1.6, label=f"threshold = {threshold:.3f}")
    ax.set_title("(query, negative) cosine similarity")
    ax.set_xlabel("stock bge-m3 cosine")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=8)

    reuse = pairs.groupby("neg_id")["query_id"].nunique()
    ax = axes[1]
    ax.hist(reuse, bins=range(1, int(reuse.max()) + 2), color=PALETTE["secondary"], edgecolor="none")
    ax.set_yscale("log")
    ax.set_title("Negative reuse across distinct queries")
    ax.set_xlabel("distinct queries served by one passage")
    ax.set_ylabel("passages (log)")

    ax = axes[2]
    top = reuse.sort_values(ascending=False).head(15)
    ax.scatter(range(len(top)), top.values, color=PALETTE["warn"], s=28)
    ax.set_title("15 most-reused negatives")
    ax.set_xlabel("rank")
    ax.set_ylabel("distinct queries")

    fig.suptitle("False negatives and generic filler (Issue 2)", y=1.02, fontsize=12)
    return _save(fig, out, dpi)


def fig_query_types(df: pd.DataFrame, out: Path, dpi: int = 300) -> Path:
    """Keyword-style vs. natural-question breakdown."""
    _style()
    types = df["query"].map(classify_query_type)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    counts = types.value_counts()
    ax = axes[0]
    b = ax.bar(counts.index, counts.values, color=[PALETTE["primary"], PALETTE["accent"]])
    ax.bar_label(b, labels=[f"{v:,}\n{100*v/len(df):.1f}%" for v in counts.values], fontsize=9)
    ax.set_title("Query style")
    ax.set_ylabel("queries")
    ax.set_ylim(0, counts.max() * 1.25)

    ax = axes[1]
    for name, color in (("keyword", PALETTE["primary"]), ("question", PALETTE["accent"])):
        sub = df.loc[types == name, "query"].map(count_tokens)
        if len(sub):
            ax.hist(sub, bins=40, alpha=0.65, label=f"{name} (median {sub.median():.0f})", color=color)
    ax.set_title("Length by query style")
    ax.set_xlabel("tokens")
    ax.legend(fontsize=8)

    fig.suptitle("Query typology", y=1.02, fontsize=12)
    return _save(fig, out, dpi)


def fig_lexical_overlap(df: pd.DataFrame, out: Path, dpi: int = 300) -> Path:
    """Query-positive token overlap — motivates carrying BM25 as a baseline."""
    _style()
    overlap = [lexical_overlap(q, p) for q, p in zip(df["query"], df["pos"])]
    overlap = np.asarray(overlap)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    ax.hist(overlap, bins=50, color=PALETTE["primary"], edgecolor="none")
    ax.axvline(overlap.mean(), color="black", lw=1, ls="--", label=f"mean {overlap.mean():.2f}")
    ax.set_title("Query tokens present in the positive passage")
    ax.set_xlabel("fraction of query tokens matched")
    ax.set_ylabel("queries")
    ax.legend(fontsize=8)

    ax = axes[1]
    types = df["query"].map(classify_query_type)
    for name, color in (("keyword", PALETTE["primary"]), ("question", PALETTE["accent"])):
        sub = overlap[(types == name).to_numpy()]
        if len(sub):
            ax.hist(sub, bins=40, alpha=0.65, label=f"{name} (mean {sub.mean():.2f})", color=color)
    ax.set_title("Overlap by query style")
    ax.set_xlabel("fraction matched")
    ax.legend(fontsize=8)

    fig.suptitle("Lexical overlap between query and positive passage", y=1.02, fontsize=12)
    return _save(fig, out, dpi)


def fig_specialty(corpus: pd.DataFrame, vocab: ClinicalVocab, out: Path, dpi: int = 300,
                  sample: int | None = None) -> tuple[Path, dict]:
    """Corpus topical distribution by MeSH disease/anatomy subcategory."""
    _style()
    texts = corpus["text"]
    if sample and sample < len(texts):
        texts = texts.sample(sample, random_state=42)

    counter: collections.Counter = collections.Counter()
    n_uncovered = 0
    for text in texts:
        subs = vocab.disease_subcategories(str(text))
        if not subs:
            n_uncovered += 1
            continue
        for s in subs:
            counter[s] += 1

    labeled = {DISEASE_SUBCATEGORIES.get(k, k): v for k, v in counter.items() if k.startswith("C")}
    top = dict(sorted(labeled.items(), key=lambda kv: kv[1], reverse=True)[:18])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(list(top)[::-1], list(top.values())[::-1], color=PALETTE["primary"])
    ax.set_xlabel("passages mentioning the category")
    ax.set_title(
        f"Corpus topical coverage by MeSH disease category\n"
        f"n={len(texts):,} passages; {n_uncovered:,} with no MeSH disease match"
    )
    stats = {
        "n_passages_scored": int(len(texts)),
        "n_no_mesh_disease_match": int(n_uncovered),
        "counts": {k: int(v) for k, v in sorted(labeled.items(), key=lambda kv: -kv[1])},
    }
    return _save(fig, out, dpi), stats
