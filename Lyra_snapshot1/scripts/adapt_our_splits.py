#!/usr/bin/env python
"""One-off adapter: rewrite this project's own cleaned splits (written by
notebooks/02_data_cleaning-2.ipynb, schema {query_id, query, pos_id, pos,
alt_pos_ids, neg_ids, negs, neg_ids_uncapped}) into the schema
lyra_capstone.models.biencoder expects ({query_id, query, split, pos_id, pos,
neg_id, neg} per query row; {id, text, split} for every corpus passage).

We are deliberately keeping our own EDA/cleaning results (a different
anaphoric filter and false-negative threshold than this package's Phase 1/2)
rather than re-running Phase 0-2 here. This script only reshapes the already-
cleaned data so Phase 3's training code can read it; it does not change which
rows survived cleaning or which negatives were kept.

Multi-positive consolidation (alt_pos_ids) has no equivalent in this schema,
which assumes one positive per query row -- we keep only the row's primary
pos_id/pos and drop the alternates, since biencoder.py's IR evaluator takes a
single pos_id as ground truth per query.
"""

import json
from pathlib import Path

OUR_PROCESSED = Path("../../data/processed")
LYRA_PROCESSED = Path(__file__).resolve().parents[1] / "artifacts" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):,} rows -> {path}")


def adapt_query_split(records: list[dict], split_name: str) -> list[dict]:
    out = []
    for r in records:
        out.append({
            "query_id": r["query_id"],
            "query": r["query"],
            "split": split_name,
            "pos_id": r["pos_id"],
            "pos": r["pos"],
            "neg_id": r["neg_ids"],
            "neg": r["negs"],
        })
    return out


def main():
    train = adapt_query_split(load_jsonl(OUR_PROCESSED / "train.jsonl"), "train")
    val = adapt_query_split(load_jsonl(OUR_PROCESSED / "val.jsonl"), "val")
    test = adapt_query_split(load_jsonl(OUR_PROCESSED / "test.jsonl"), "test")

    corpus_rows = load_jsonl(OUR_PROCESSED / "corpus_clean.jsonl")
    corpus_out = [
        {"id": r["id"], "text": r["text"], "split": r["split"]} for r in corpus_rows
    ]

    write_jsonl(LYRA_PROCESSED / "train.jsonl", train)
    write_jsonl(LYRA_PROCESSED / "val.jsonl", val)
    write_jsonl(LYRA_PROCESSED / "test.jsonl", test)
    write_jsonl(LYRA_PROCESSED / "corpus.jsonl", corpus_out)


if __name__ == "__main__":
    main()
