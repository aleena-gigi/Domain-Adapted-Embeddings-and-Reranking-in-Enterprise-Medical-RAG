"""Phase 1 — audit and filter (spec detail §1.1–§1.3).

Three data-quality issues from spec v2 §2.2:
  1. anaphoric / context-dependent queries  (D5 — removed from train and eval)
  2. false negatives                        (dropped, rate reported)
  3. degenerate positives                   (dropped, count reported)

Every filter returns evidence alongside its verdict, because the removal rates
are themselves EDA deliverables for paper Section 2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .vocab import ClinicalVocab

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1.1 Anaphoric query filter (D5 — load-bearing)
# --------------------------------------------------------------------------

# A bare definite reference to the case subject. Each pattern is anchored on a
# *definite* determiner or a possessive pronoun: "the patient", "this patient",
# "her stay". Indefinite mentions ("a patient with TEVAR") are not anaphoric —
# they describe a class, not one unresolved case.
DEFINITE_REF_PATTERNS: tuple[tuple[str, str], ...] = (
    ("the_patient", r"\bthe\s+patient(?:'s|’s|s'|s’)?\b"),
    ("this_patient", r"\b(?:this|that|our)\s+patient(?:'s|’s)?\b"),
    ("bare_patient_poss", r"^\s*patient(?:'s|’s)\b"),
    ("the_case_subject", r"\bthe\s+(?:man|woman|boy|girl|infant|baby|neonate|child)\b"),
    ("poss_pronoun", r"\b(?:his|her|their)\s+[a-z]+"),
    ("subject_pronoun", r"\b(?:he|she)\s+(?:was|were|is|had|has|did|received|underwent)\b"),
)

_COMPILED_REFS = tuple((name, re.compile(p, re.I)) for name, p in DEFINITE_REF_PATTERNS)


@dataclass
class AnaphoricFilterResult:
    is_anaphoric: bool
    matched_pattern: str | None
    clinical_terms_found: list[str] = field(default_factory=list)


def is_anaphoric_query(query: str, clinical_vocab: ClinicalVocab) -> AnaphoricFilterResult:
    """Flag context-dependent queries with no unique correct answer.

    Positive examples (drop these):
        "What was the outcome of the patient's treatment?"
        "What was the patient's condition at birth?"
        "How did the patient fare after the surgery?"

    Negative examples (keep these):
        "thyroid nodule symptoms"
        "What NIHSS score indicates severe stroke?"
        "TEVAR postoperative complications"

    Rule: flag if the query contains a bare definite reference to the patient
    AND contains no term from `clinical_vocab`.

    The second clause is what saves "What was the patient's NIHSS score?" —
    anaphoric in form, but the clinical entity makes it retrievable. It is also
    what saves "Role of ECMO in the patient's treatment" and "What were the
    results of the patient's follow-up MRI?", which are common in this corpus.
    """
    matched = next((name for name, rx in _COMPILED_REFS if rx.search(query)), None)
    # Terms are computed unconditionally, not short-circuited on `matched`:
    # the EDA reports clinical-entity density across the whole query set, and
    # `is_generic_no_entity` is meaningless if unmatched rows silently carry an
    # empty list.
    terms = clinical_vocab.find(query)
    return AnaphoricFilterResult(
        is_anaphoric=matched is not None and not terms,
        matched_pattern=matched,
        clinical_terms_found=terms,
    )


def apply_anaphoric_filter(df: pd.DataFrame, clinical_vocab: ClinicalVocab) -> pd.DataFrame:
    """Add is_anaphoric / matched_pattern / clinical_terms / n_clinical_terms."""
    results = [is_anaphoric_query(q, clinical_vocab) for q in df["query"]]
    out = df.copy()
    out["is_anaphoric"] = [r.is_anaphoric for r in results]
    out["matched_pattern"] = [r.matched_pattern for r in results]
    out["clinical_terms"] = [r.clinical_terms_found for r in results]
    out["n_clinical_terms"] = [len(r.clinical_terms_found) for r in results]
    # Secondary signal, reported but NOT dropped: queries naming no clinical
    # entity at all, even without a patient reference ("Disease progression
    # monitoring"). Same defect — no unique gold passage — but outside D5's
    # stated rule, so it is surfaced for the write-up rather than acted on.
    out["is_generic_no_entity"] = (out["n_clinical_terms"] == 0) & ~out["is_anaphoric"]
    return out


def validate_anaphoric_filter(
    df: pd.DataFrame, n_sample: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Emit a stratified sample for hand-labeling.

    50 flagged + 50 unflagged, shuffled, with a blank `human_label` column.
    The team fills this in manually. Filter precision and recall computed from
    it go in the paper — an unvalidated filter that removes a third of the
    dataset will not survive review.
    """
    half = n_sample // 2
    flagged = df[df["is_anaphoric"]]
    unflagged = df[~df["is_anaphoric"]]

    n_flag = min(half, len(flagged))
    n_unflag = min(n_sample - n_flag, len(unflagged))
    if n_flag < half:
        log.warning("only %d flagged queries available for the %d-sample", n_flag, n_sample)

    sample = pd.concat(
        [
            flagged.sample(n_flag, random_state=seed),
            unflagged.sample(n_unflag, random_state=seed),
        ]
    ).sample(frac=1.0, random_state=seed)

    out = sample[["query_id", "query", "is_anaphoric", "matched_pattern", "clinical_terms"]].copy()
    out = out.rename(columns={"is_anaphoric": "filter_label"})
    out["clinical_terms"] = out["clinical_terms"].apply(lambda t: "; ".join(t))
    # Blank column for the human annotator: 1 = anaphoric (unanswerable in
    # isolation), 0 = answerable. Do NOT show the annotator filter_label first.
    out["human_label"] = ""
    out["annotator_note"] = ""
    return out.reset_index(drop=True)


def score_filter_agreement(labeled: pd.DataFrame) -> dict:
    """Precision / recall / F1 / accuracy of the filter against hand labels.

    Treats "anaphoric" as the positive class. Called once the annotator has
    filled `human_label` in the validation sample.
    """
    df = labeled.dropna(subset=["human_label"])
    df = df[df["human_label"].astype(str).str.strip() != ""]
    if df.empty:
        raise ValueError("no human labels present — fill the human_label column first")

    truth = df["human_label"].astype(str).str.strip().isin({"1", "true", "True", "yes", "y"})
    pred = df["filter_label"].astype(bool)

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "n_labeled": int(len(df)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(df), 4),
    }


# --------------------------------------------------------------------------
# 1.2 False-negative detection
# --------------------------------------------------------------------------


def score_negative_plausibility(
    df: pd.DataFrame,
    corpus: dict[str, str],
    model_name: str = "BAAI/bge-m3",
    device: str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Score every (query, negative) pair with the STOCK encoder.

    Stock, not fine-tuned — this runs before training and must not depend on it.

    Returns one row per (query_id, neg_id) with `neg_similarity` and
    `neg_reuse_count` (how many distinct queries share this neg_id). High reuse
    indicates generic filler like "The patient was provided with adequate
    postoperative care."
    """
    from sentence_transformers import SentenceTransformer

    from ..devices import infer_device

    device = device or infer_device()

    pairs = df[["query_id", "query", "neg_id"]].explode("neg_id").dropna(subset=["neg_id"])
    pairs = pairs.reset_index(drop=True)

    reuse = pairs.groupby("neg_id")["query_id"].nunique().rename("neg_reuse_count")
    pairs = pairs.join(reuse, on="neg_id")

    model = SentenceTransformer(model_name, device=device)

    uniq_queries = df[["query_id", "query"]].drop_duplicates("query_id")
    log.info("encoding %s queries on %s", f"{len(uniq_queries):,}", device)
    q_emb = model.encode(
        uniq_queries["query"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    q_index = {qid: i for i, qid in enumerate(uniq_queries["query_id"])}

    neg_ids = pairs["neg_id"].unique().tolist()
    log.info("encoding %s unique negative passages", f"{len(neg_ids):,}")
    n_emb = model.encode(
        [corpus[i] for i in neg_ids],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    n_index = {nid: i for i, nid in enumerate(neg_ids)}

    qi = pairs["query_id"].map(q_index).to_numpy()
    ni = pairs["neg_id"].map(n_index).to_numpy()
    # Embeddings are L2-normalized, so the row-wise dot product is cosine.
    pairs["neg_similarity"] = np.einsum("ij,ij->i", q_emb[qi], n_emb[ni]).astype(float)

    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    return pairs


def negative_similarity_percentiles(pairs: pd.DataFrame) -> dict:
    """Percentiles of the (query, negative) similarity distribution.

    The threshold is chosen from this distribution, never a priori (Open Item
    #4). The audit report carries these numbers so the choice is auditable.
    """
    sims = pairs["neg_similarity"].to_numpy()
    qs = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
    return {
        "n_pairs": int(len(sims)),
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
        "percentiles": {str(q): float(np.percentile(sims, q)) for q in qs},
    }


def flag_false_negatives(
    df: pd.DataFrame, pairs: pd.DataFrame, sim_threshold: float
) -> tuple[pd.DataFrame, dict]:
    """Drop negatives above threshold.

    Threshold is set from the similarity distribution, not picked a priori —
    inspect the histogram first and record the chosen value in config.

    Returns the frame with pruned `neg`/`neg_id` lists, plus a stats dict.
    """
    flagged = pairs[pairs["neg_similarity"] >= sim_threshold]
    dropped: dict[str, set[str]] = (
        flagged.groupby("query_id")["neg_id"].agg(set).to_dict()
    )

    def prune(row):
        drop = dropped.get(row["query_id"])
        if not drop:
            return row["neg"], row["neg_id"]
        keep = [(t, i) for t, i in zip(row["neg"], row["neg_id"]) if i not in drop]
        if not keep:
            return [], []
        texts, ids = zip(*keep)
        return list(texts), list(ids)

    out = df.copy()
    pruned = out.apply(prune, axis=1, result_type="expand")
    out["neg"], out["neg_id"] = pruned[0], pruned[1]
    out["n_negatives"] = out["neg_id"].apply(len)

    stats = {
        "sim_threshold": float(sim_threshold),
        "pairs_total": int(len(pairs)),
        "pairs_flagged": int(len(flagged)),
        "pairs_flagged_rate": round(len(flagged) / len(pairs), 6) if len(pairs) else 0.0,
        "queries_affected": int(len(dropped)),
        "queries_left_with_zero_negatives": int((out["n_negatives"] == 0).sum()),
        "mean_negatives_after": round(float(out["n_negatives"].mean()), 3),
    }
    return out, stats


# --------------------------------------------------------------------------
# 1.3 Degenerate positive filter
# --------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", str(text)))


def filter_degenerate_positives(df: pd.DataFrame, min_tokens: int) -> tuple[pd.DataFrame, dict]:
    """Drop near-contentless positives ("Nurick scale 1.").

    Passages range 5-993 chars; the low tail is unusable as a retrieval target.
    Reports the count removed.
    """
    tokens = df["pos"].map(count_tokens)
    keep = tokens >= min_tokens
    removed = df.loc[~keep]

    stats = {
        "min_tokens": int(min_tokens),
        "n_before": int(len(df)),
        "n_removed": int((~keep).sum()),
        "removal_rate": round(float((~keep).mean()), 6),
        "n_after": int(keep.sum()),
        "examples_removed": removed["pos"].head(10).tolist(),
    }
    return df.loc[keep].copy(), stats


# --------------------------------------------------------------------------
# Query typology (EDA deliverable: keyword-style vs. natural question)
# --------------------------------------------------------------------------

_QUESTION_WORDS = (
    "what", "how", "why", "when", "where", "which", "who", "is", "are",
    "was", "were", "does", "do", "did", "can", "should", "would", "will",
)


def classify_query_type(query: str) -> str:
    """keyword | question — spec v2 §2.4 asks for this breakdown."""
    q = query.strip()
    if q.endswith("?"):
        return "question"
    return "question" if q.lower().split()[:1] and q.lower().split()[0] in _QUESTION_WORDS else "keyword"


def lexical_overlap(query: str, passage: str) -> float:
    """Fraction of query tokens present in the passage.

    Proxy for how much of the task is solvable by exact matching — this is what
    motivates carrying BM25 as a baseline (spec v2 §2.4).
    """
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q:
        return 0.0
    p = set(re.findall(r"[a-z0-9]+", str(passage).lower()))
    return len(q & p) / len(q)
