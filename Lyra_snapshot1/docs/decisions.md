# Decisions Log

Resolutions to the Open Items in `capstone_implementation_spec_v2.md` §10 /
`implementation_spec_detail.md` §12, plus deviations forced by the execution
environment. Every entry records the evidence, not just the choice.

---

## E1 — Execution environment deviates from spec §0.3: one GPU, not two

**Spec assumed:** 48 GB (training) + 12 GB (inference/embedding/judging), running
concurrently.

**Actual:** a single NVIDIA RTX A6000, 49,140 MiB. No second card.

**Consequence.** Every `device: str = "cuda:1"` in the spec's signatures resolves
to the one device, through `src/lyra_capstone/devices.py` so the assumption
lives in exactly one file. The workloads §11 ran in parallel are sequenced:

| §11 claim | Status |
|---|---|
| BM25 / MedEmbed index builds during Phase 3 training | Sequenced after training |
| Phase 9 judge validation during Phase 8 generation | Judge runs after generation |
| Phase 1 hand-labeling during automated filtering | Unaffected — human work |
| Section 2 write-up once Phase 1 lands | Unaffected |

There is also no CI runner configured on this project, so Phase 2's
"the assertion is in CI, not just a notebook" criterion is met by putting the
leakage audit in two executable places: `scripts/02_build_splits.py`, which
raises and aborts the build, and `tests/test_splits.py`, which re-derives it
from the written artifacts under `pytest`.

Cost is small: the parallelizable stages are the cheap ones. Phase 1's full
GPU pass (198,257 query-negative pairs) took **under two minutes**. Nothing
else in the plan depends on the second card.

---

## O1 — Milvus vs. FAISS for dense indexes → **Milvus**

`milvusdb/milvus:v2.6.16` is already running in Docker on this box
(`milvus-standalone`, with `milvus-etcd` and `milvus-minio`). Standing it up
costs nothing, and it preserves the LLD parity argument — the eval path uses
the same index technology as the production Lyra system, so the 1024-dim COSINE
collection contract is exercised rather than simulated. Revisit only if index
build time becomes a bottleneck, which is implausible at 27,590 passages.

---

## O2 — Clinical vocabulary: UMLS/MeSH term list vs. scispacy → **MeSH**

The spec says decide on install friction, not marginal accuracy.

- **MeSH:** one 30 MB public-domain file (`d2025.bin`) from NLM. No
  registration, no license click-through, no dependency.
- **scispacy:** pins an older spaCy and pulls a model wheel from a third-party
  bucket; on Python 3.11 with numpy 2.x this is a live conflict risk.
- **UMLS:** requires a UTS account and a license agreement. Rejected on friction.

MeSH pays a second dividend: `MN` tree numbers give the corpus
specialty/body-system distribution (`eda_specialty.png`) from the same file, so
one download serves two deliverables.

**Two implementation findings that materially affect the filter:**

1. **Entry terms are not optional.** The tree file (`mtrees2025.bin`) carries
   only *preferred* headings. Using it alone, only 62.4% of queries matched any
   clinical term, and answerable queries were dropped because "Quadriparesis"
   and "Chemotherapy" are entry terms whose preferred headings are
   "Quadriplegia" and "Drug Therapy". Switching to the descriptor file and
   indexing `ENTRY` / `PRINT ENTRY` synonyms lifted coverage to **77.1%**.

2. **A blocklist is required, or the filter's second clause self-destructs.**
   MeSH contains "Patients", "Pain", "Surgery", "Treatment Outcome" and
   "Patient Discharge" as descriptors — precisely the words anaphoric queries
   are built from. Without blocking them, "What was the outcome of the
   patient's treatment?" counts as naming a clinical entity and survives.
   `GENERIC_MESH_TERMS` (125 entries) is derived from the highest-frequency
   content words among queries carrying a definite patient reference, and is
   frozen in `vocab.py` so the filter is a fixed function of its input rather
   than something refit per run. Blocking a descriptor also blocks its
   synonyms, otherwise blocking "Drug Therapy" leaks back in as
   "Pharmacotherapy".

MeSH indexes almost no named instruments, so an eponym-measure pattern
(`Nurick scale`, `Cobb angle`, `Clavien-Dindo grade`) supplements it. These are
the exact vocabulary the domain adaptation is meant to learn (spec v2 §1).

---

## O4 — Negative-similarity threshold → **0.71 (p99)**

Set from the observed distribution, as the spec requires — never a priori.

198,257 (query, negative) pairs scored with stock bge-m3:
mean 0.521, sd 0.074, min 0.226, max 0.910. The histogram is **smooth and
unimodal with no elbow**, so the value was chosen by reading pairs banded by
similarity:

| Band | What is actually in it |
|---|---|
| 0.62–0.66 | Mixed. Contains genuine false negatives *and* fair negatives |
| 0.70–0.74 | Near-uniformly generic follow-up boilerplate |
| 0.78+ | Unambiguous false negatives |

At 0.910: query *"Why was the patient advised to follow up with her primary
care physician?"* paired against negative *"She was advised to follow up with
her primary care provider."*

Everything above 0.71 is harmful as a negative — it is either a true false
negative or contentless filler. Effect: **1,962 pairs dropped (0.99%)**, 766
queries affected, mean negatives per query 10.36 → 10.26. Ten queries are left
with zero explicit negatives; they still train on in-batch negatives under
`CachedMultipleNegativesRankingLoss`.

**Caveat to report in the paper.** This filter's *recall* is limited. Real
false negatives persist below the threshold — a heart-failure/polyneuropathy
pair that is plainly relevant scores only 0.633, mid-distribution. Cosine
similarity separates *generic* from *specific* far better than it separates
*relevant* from *irrelevant*. The reuse statistic is the better-targeted signal
for filler, which is why both distributions are reported.

---

## P2a — Cross-split near-duplicate threshold → **0.95 cosine**

Not an Open Item; it fell out of building Phase 2. The spec requires
`audit_leakage` to find zero "duplicate queries straddling splits (exact +
near-dup)" but does not say what near-dup means, and the answer is not
guessable — it has to come from the corpus.

Every query's nearest neighbour *in a different split*, stock bge-m3, same-split
candidates masked: 17,051 pairs, mean 0.775, max 0.999. Read in bands:

| Band | What is in it |
|---|---|
| ≥ 0.95 | Paraphrases and case variants of one information need |
| 0.93–0.95 | Mixed: real twins beside genuinely distinct scenarios |

The cut has to reach lower than intuition suggests. bge-m3 scores
*"Post-operative care plan for Solitary Fibrous Tumor"* against its own
lowercase form at only **0.934**, because these queries are short and casing
moves the embedding. Meanwhile 0.93–0.95 also holds *"small bowel obstruction"*
vs *"small bowel resection"* — different procedures, not a duplicate.

0.95 flags 503 pairs and costs 463 train + 22 val queries, 3.0% of train.
Exact (normalized) duplicates are handled separately and cost 100 more.
Full pair list preserved at `artifacts/interim/near_duplicate_pairs.parquet`.

**Precedence, not deletion.** A straddling twin is resolved by dropping the copy
in the lower-precedence split, **test > val > train**: test is the reported
number and is never modified, val is the model-selection instrument and
outranks train. Reassigning a query to a different split is not an option — it
would break the "query inherits its `pos_id`'s partition" invariant, which is
the whole basis of the leakage guarantee. Zero test queries are dropped at any
threshold, by construction.

**Reportable side effect.** These twins point at *different* passages by
construction (different partitions). So the test set contains queries whose
train paraphrase has a different gold passage — which means some test queries
almost certainly have a second valid answer that scoring will count as a miss.
That understates absolute recall for every configuration equally, so it does
not threaten the comparison, but the absolute numbers should be read with it in
mind.

---

## P2b — Phase 2 re-mining is capped at Phase 1's false-negative threshold

`sanitize_training_negatives` strips any negative pointing into val/test
(30,217 of 151,463) and re-mines replacements from the train partition by stock
bge-m3 top-k. Top-k against the query is *exactly* where false negatives live,
so mining uncapped would quietly reintroduce the pairs Phase 1 removed. The cap
inherits `false_negatives.sim_threshold` (0.71) and rejected 11,019 candidates.

Rows are padded **and truncated** to 8 negatives so Phase 3 gets a fixed
(anchor, positive, negative_1..8) schema: 14,830 of 14,859 train rows hit
exactly 8. The 29 that fall short are all generic follow-up queries
(*"What follow-up care was recommended?"*) whose entire top-128 sits above the
cap — the same population Phase 1 flagged, arriving by a different route. They
still train on in-batch negatives under `CachedMultipleNegativesRankingLoss`.

This is *not* the hard-negative mining of Phase 4, which re-mines from the
**fine-tuned** retriever and is deliberately more aggressive. Phase 2's job is
only to make the training set legal and rectangular.

---

## Open

| # | Item | Blocks | Status |
|---|---|---|---|
| O3 | NLI judge checkpoint + license | Phase 9 | Proposed: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MIT, ~2 GB). Not yet confirmed. |
| O5 | Forgetting check in paper or appendix | Phase 10 | Undecided; low cost either way |
| — | Hand-validation of the anaphoric filter | Phase 1 sign-off | **Outstanding.** Sample written; needs 100 human labels |
| — | Fallback if filtering ever drops below 8,000 queries | Phase 1 | Not needed: 19,141 survive. Contingency unwritten |

---

## Spec amendments proposed

**`is_generic_no_entity` is a coverage statistic, not a defect class.** Phase 1
reports 2,753 queries that carry no patient reference *and* name no MeSH
entity. It is tempting to read these as a second unanswerable class, but
inspection says otherwise: "perforating dermatosis symptoms" and "Schwannoma
gastrointestinal treatment" are perfectly answerable — MeSH simply does not
index those surface forms. The number measures vocabulary coverage and bounds
how much the filter's second clause can save. It is reported, **not dropped**,
and D5's stated rule governs what is removed.
