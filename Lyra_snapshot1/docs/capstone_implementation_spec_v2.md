# Capstone Implementation Spec v2 — Domain-Adapted Retrieval for Clinical-Note RAG

**Supersedes** §2–§8 of `prelim_planning`. §1 (rationale) and §7 (LLD seam) of the original remain valid with the amendments in §7 below.

**Status of change:** the original plan assumed a self-built synthetic corpus from PMC full text. The project now uses the pre-existing MedEmbed triplet dataset, which eliminates corpus construction, synthetic pair generation, and the non-expert spot-check burden. This spec adapts every downstream decision to that substitution and to the resulting domain re-scope.

---

## 0. Decision Record

| # | Decision | Rationale |
|---|---|---|
| D1 | Scope is **clinical-note retrieval**, not consumer-health or abstract-level | MedEmbed source text is PMC clinical notes / discharge summaries / case reports |
| D2 | **Both** models are trained: fine-tuned bi-encoder (A) and fine-tuned cross-encoder (B) | Restores Exp B from the original plan; gives two learned components for a 3-person capstone |
| D3 | Primary eval is a **held-out in-domain slice**; NFCorpus is the **out-of-domain transfer check** | NFCorpus is abstract-level consumer health — no longer in-domain after D1 |
| D4 | BioASQ and TREC-COVID are **dropped** | Removes BioASQ licensing/registration friction and TREC-COVID's 171k-doc index cost; neither is in-domain after D1 |
| D5 | Anaphoric queries are **removed from both train and eval** | They have no unique correct answer; they are noise in training and invalid as eval items |
| D6 | Results table includes **BM25 and MedEmbed-large-v0.1** as reference baselines | BM25 is a strong lexical baseline on clinical text; MedEmbed is prior art on the same data |
| D7 | Faithfulness judged by an **NLI model**, not by an LLM | Avoids the generator judging itself |

---

## 1. Scope and Framing

The system adapts the retrieval stack of an existing enterprise RAG application (Lyra) to **clinical documentation**: discharge summaries, case reports, and clinical narrative text. The generator (gemma4) is frozen throughout.

The domain argument for why generic embeddings underperform is now sharper than the original consumer-health framing. Clinical notes are dense with abbreviations and instrument names that carry no meaning in general-purpose embedding space — TEVAR, NIHSS, Nurick scale, EBV placement, ANBP, Clavien-Dindo grades. A model trained on web text has no representation for these, so lexical near-misses dominate dense retrieval. This is the mechanism the fine-tune is meant to fix, and it is directly observable in error analysis.

End users are clinical support and knowledge-management staff querying an internal store of clinical documentation.

**Hypothesis (two parts).**
1. Domain adaptation of the retrieval stack improves Recall@k, MRR, and nDCG@10 over a stock bi-encoder, in-domain, with the trained reranker adding further gain over an off-the-shelf one.
2. That retrieval gain propagates to **answer faithfulness** despite the generator being unchanged.

Part 2 is the contribution. Part 1 is table stakes and is expected to replicate prior work.

---

## 2. Dataset

### 2.1 Ground truth about the source data

`abhinand/MedEmbed-training-triplets-v1` (Apache 2.0). Correct figures — the draft paper currently overstates these:

| Quantity | Value |
|---|---|
| Triplet rows (`default` config) | 232,684 |
| **Unique queries** | 21,689 |
| **Unique corpus passages** | 27,590 |
| Negatives per query (mean) | ~10.7 |
| Query length | 10–131 characters |
| Passage length | 5–993 characters |
| Generation | LLaMA 3.1 70B over PMC clinical notes |

The effective dataset is ~21.7k queries, not 230k independent examples. All reported figures in the paper must use the unique-query count.

The `merged` and `queries` configs (21,689 rows each) are the per-query views; `corpus` (27,590) is the passage store. Use `merged` for training — it gives all negatives per query in one row.

### 2.2 Known data quality issues (this is the Section 2 EDA)

**Issue 1 — anaphoric / context-dependent queries.** A substantial share of queries are unanswerable in isolation: "What was the outcome of the patient's treatment?", "What was the patient's condition at birth?", "How did the patient fare after the surgery?". Thousands of near-identical queries point at different passages, so no unique gold passage exists.

*Handling (D5):* remove from train and eval. Detection rule: query contains a definite reference to the patient (`the patient`, `the patient's`, `his/her` + generic noun) **and** contains no clinical entity term (matched against a UMLS/MeSH term list or a clinical NER pass). Validate the rule by hand-labeling 100 sampled classifications; report filter precision and recall in the paper. Report the removal rate — it is a headline EDA number.

**Issue 2 — false negatives.** Generic passages recur as negatives across unrelated queries (e.g. "The patient was provided with adequate postoperative care." appears as `neg` for multiple distinct queries). Some negatives are as relevant as their paired positive.

*Handling:* score every (query, neg) pair with stock bge-m3; flag pairs above a similarity threshold. Separately, count distinct queries served by each `neg_id` — high-frequency negatives are generic filler. Report both distributions. Drop flagged pairs before training; keep the rate as a reported quality signal.

**Issue 3 — degenerate positives.** Some positives are near-contentless ("Nurick scale 1.", "The patient was treated for his symptoms while in the hospital."). Filter positives below a minimum token count and report the count removed.

### 2.3 Splits

Partition the **27,590 corpus passages** 80/10/10 into train/val/test partitions. Assign each query to the partition containing its `pos_id`.

Then, within training data only, discard any negative whose `neg_id` falls in the val or test partition, and re-mine replacements from the train partition using stock bge-m3 top-k. This guarantees no test-partition passage is ever seen during training in any role.

At evaluation time, index the **full 27,590-passage corpus** for every configuration so the distractor pool is identical across the table. Only test-partition queries are scored.

### 2.4 EDA deliverables for the paper

- Query length and passage length distributions
- Anaphoric-query rate, with filter validation numbers
- False-negative rate and negative-reuse frequency distribution
- Query type breakdown: keyword-style ("thyroid nodule symptoms") vs. natural-question style
- Lexical overlap between query and positive passage (proxy for how much of the task is solvable by exact matching — motivates the BM25 baseline)
- Corpus topical distribution (specialty / body system), to show coverage breadth

---

## 3. Experiment A — Fine-Tuned Bi-Encoder

**Base model:** `BAAI/bge-m3` (XLM-R-large scale, ~568M params, 1024-dim dense output). Output dimension and COSINE metric are unchanged, so the existing Milvus collection contract holds.

**Objective:** `CachedMultipleNegativesRankingLoss` (GradCache) in `sentence-transformers` — combines in-batch negatives with the mined hard negatives, and the caching lets the effective batch far exceed what fits in memory. Large batches matter disproportionately for contrastive retrieval training.

**Configuration:**

| Parameter | Value |
|---|---|
| `max_seq_length` | 320 tokens (passages cap at 993 chars — 8192 is wasted memory) |
| Effective batch | 256+ via GradCache; mini-batch sized to fit 48 GB |
| Precision | bf16, gradient checkpointing on |
| Learning rate | 1e-5 to 2e-5 |
| Epochs | 1–3, early stop on val nDCG@10 |
| Warmup | 10% |
| Hard negatives per query | 4–8 (swept) |
| Normalization | L2, COSINE at train and serve |

At ~21.7k queries (minus filtering) an epoch is a few hundred steps — minutes on the 48 GB card. **Use that headroom for a real sweep** over learning rate × epochs × negatives-per-query. The template's "how did you optimize your model" section needs actual comparative content, and this is where it comes from.

**Serving:** re-embed the corpus with the fine-tuned model into a parallel `_ft` collection. Probe dimension at load from a test embedding; never hardcode 1024.

---

## 4. Experiment B — Fine-Tuned Cross-Encoder

**Base model:** `BAAI/bge-reranker-v2-m3`.

**Training data — do not reuse MedEmbed's negatives.** Run the *fine-tuned* retriever from Experiment A over the train split, take top-N (N = 30–50) non-positive results, and use those as negatives. The reranker must train on the distribution it will actually see at inference; that is the entire reason it exists.

| Parameter | Value |
|---|---|
| Objective | Binary cross-entropy on (query, passage) relevance |
| Negatives per positive | ~8 |
| Learning rate | 2e-5 |
| Epochs | 1–2 |
| Early stopping | Aggressive, on val nDCG@10 |

Cross-encoders converge fast and overfit faster. If val nDCG turns over mid-epoch, stop there.

**Pipeline position:** after Milvus filtered search, before parent expansion. Reranking sits inside the access/scope filter, so isolation guarantees are untouched. Latency at N≈30–50 is tens of milliseconds and belongs to the chat GPU lease, not a separate low-priority request.

---

## 5. Evaluation

### 5.1 Configuration matrix

| # | Retriever | Reranker | Role |
|---|---|---|---|
| 1 | BM25 | none | Lexical reference — may beat config 2 |
| 2 | Stock bge-m3 | none | Primary baseline |
| 3 | MedEmbed-large-v0.1 | none | Prior art, same training data, 1024-dim |
| 4 | Fine-tuned bge-m3 | none | Experiment A |
| 5 | Fine-tuned bge-m3 | stock bge-reranker-v2-m3 | Isolates "reranking helps" |
| 6 | Fine-tuned bge-m3 | fine-tuned reranker | Experiment B — full system |

Config 5 is the ablation that justifies training the reranker at all. Without it, the marginal value of Experiment B is unmeasured.

### 5.2 Test tiers

**Tier 1 — in-domain (primary).** Held-out test partition of the MedEmbed clinical corpus, post-filtering. Full 27,590-passage index. All six configs.

**Tier 2 — out-of-domain transfer (credibility anchor).** NFCorpus: 3,633 documents, 323 test queries. All six configs. Cheap to index and genuinely external.

Tier 2 exists to answer "does clinical-note adaptation transfer to abstract-level medical retrieval, or is it narrow?" Either answer is a legitimate result. State both tiers side by side; do not bury a gap.

*Optional, high value:* one non-medical BEIR slice against config 4 to check for catastrophic forgetting. If domain adaptation costs general retrieval ability, that is exactly the "unexpected result" the conclusion section asks for.

### 5.3 Metrics

- Recall@{1,5,10}
- MRR
- nDCG@10
- Paired bootstrap over per-query scores for significance between adjacent configs

---

## 6. Faithfulness Harness

Generation runs with gemma4 frozen and identical prompting across all configs. Only the retrieved context differs.

**Judge:** DeBERTa-v3-large-MNLI class model. Sentence-split each generated answer into claims; for each claim, run entailment against the retrieved context as premise; faithfulness = fraction of claims entailed.

**Two mechanical constraints:**
- NLI models cap at 512 tokens. Chunk the premise and take max entailment per claim.
- The entailment threshold is a free parameter. Validate against ~50 human-labeled claim/context pairs and report the agreement rate. An unvalidated judge cannot carry the headline claim.

**Report four numbers per config, never faithfulness alone:**

| Metric | Why |
|---|---|
| Faithfulness | The headline |
| Answer relevance | Does the answer address the question at all |
| Context precision | How much retrieved context was actually useful |
| **Abstention rate** | The critical confound |

Abstention is the trap. A model that answers "the context does not contain this" is trivially 100% faithful. If faithfulness rises *only* because abstention rises, the result is null dressed as a win. If abstention falls while faithfulness rises, the causal story is clean. Track it from the first run, not as an afterthought.

**Sample:** 300–500 test queries per config. Generation is the slow step and full coverage adds nothing.

Run the harness on both tiers. Tier 1 carries the claim; Tier 2 tests whether it survives a genre shift.

---

## 7. LLD Integration (amendments to prelim_planning §7)

Unchanged from the original plan: 1024-dim COSINE contract, runtime dimension probe, parallel `_ft` collection for A/B, rerank stage inside the access filter, chat-lease GPU accounting, frozen gemma4.

Amended:
- Corpus is the MedEmbed 27,590-passage store, not a self-built PMC crawl. No parent-child chunking is required — passages are already short and atomic. **This removes the parent-expansion step from the eval path**, though it remains in the production system for uploaded documents.
- Training is fully offline and never contends with the serving GPU schedule.

---

## 8. Schedule (~6 weeks + 2 buffer)

| Week | Milestone |
|---|---|
| 1 | Data audit, filter implementation + hand validation, splits, all EDA figures |
| 2 | Experiment A training + hyperparameter sweep |
| 3 | Configs 1–4 evaluated on both tiers; baselines locked |
| 4 | Negative mining from fine-tuned retriever; reranker training |
| 5 | Configs 5–6; full six-row table; faithfulness harness on both tiers |
| 6 | Error analysis, write-up |
| 7–8 | Buffer |

---

## 9. Required Edits to the Draft Paper

**Abstract.** Written last.

**Introduction, paragraph 1.** Soften the novelty claim. "Prior work on domain-adapting medical retrievers has measured success purely in retrieval terms" is too absolute — there is existing work on retrieval quality and generation hallucination. Reframe as underexplored *for domain-adapted medical retrievers in an enterprise RAG setting*, and cite two examples.

**Introduction, paragraph 2.** Replace the "dyspnea vs. shortness of breath" example. That illustrates a lay/clinical vocabulary gap, which is the consumer-health framing the project no longer uses. Substitute clinical abbreviation opacity (TEVAR, NIHSS, Clavien-Dindo) — a sharper and more defensible mechanism.

**Introduction, paragraph 3.** Re-describe end users as clinical documentation / knowledge-management staff rather than consumer-health users.

**Introduction, paragraph 4.** Correct dataset figures to 21,689 queries / 27,590 passages / 232,684 triplets. Name LLaMA 3.1 70B as the generator. Remove BioASQ and TREC-COVID; describe NFCorpus as an out-of-domain transfer check rather than the primary benchmark. Remove the corresponding citations (Tsatsaronis et al.) and keep Thakur et al. for BEIR/NFCorpus.

**Introduction, paragraph 5.** Update to a six-configuration comparison with two trained models. The current text states the reranker is "used as-is with no extra training" — no longer true.

**Title.** Still accurate. Consider narrowing the subtitle to signal the clinical-note scope.

---

## 10. Open Items

- Exact filter rule for anaphoric detection: UMLS term list vs. off-the-shelf clinical NER — pick based on install friction, not accuracy.
- BM25 implementation: `rank_bm25` (simplest, adequate at 27.6k docs) vs. Pyserini.
- Whether MedEmbed-large-v0.1's 512-token limit needs any special handling at this passage length (probably not — passages cap at 993 chars).
- Confirm the NLI judge model choice and license before building the harness around it.
- Decide whether the non-medical forgetting check makes the paper or an appendix.