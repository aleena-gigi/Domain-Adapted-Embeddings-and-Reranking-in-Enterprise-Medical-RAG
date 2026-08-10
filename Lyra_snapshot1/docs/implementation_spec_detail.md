# Implementation Spec — Detail

**Project:** Domain-Adapted Embeddings and Reranking in Enterprise Clinical-Note RAG
**Companion doc:** `capstone_implementation_spec_v2.md` (decisions and rationale). This document is the *executable* contract — phases, file paths, signatures, acceptance criteria.

**Audience:** the team and coding agents. Every phase is written so an agent can be handed one phase, its inputs, and its acceptance criteria, and produce working code without needing the rest of the document.

**Execution target:** local box, 2 GPUs — 48 GB (training) and 12 GB (inference, embedding, judging). No cloud dependency.

---

## 0. Conventions

### 0.1 Repository layout

```
lyra-capstone/
├── configs/
│   ├── data.yaml                  # filter thresholds, split ratios, seeds
│   ├── train_biencoder.yaml       # Experiment A
│   ├── train_reranker.yaml        # Experiment B
│   └── eval.yaml                  # config matrix, metrics, k values
├── src/lyra_capstone/
│   ├── config.py                  # dataclass config loader
│   ├── paths.py                   # artifact registry, no hardcoded paths elsewhere
│   ├── seeds.py                   # global determinism
│   ├── data/
│   │   ├── load.py
│   │   ├── filters.py
│   │   ├── splits.py
│   │   └── build.py
│   ├── models/
│   │   ├── biencoder.py
│   │   └── reranker.py
│   ├── mining/hard_negatives.py
│   ├── indexes/
│   │   ├── dense.py               # Milvus or FAISS
│   │   └── lexical.py             # BM25
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── retrieval.py
│   │   ├── faithfulness.py
│   │   └── significance.py
│   └── generation/generate.py
├── scripts/                       # thin CLI wrappers, argparse only, no logic
│   ├── 00_download.py
│   ├── 01_audit_filter.py
│   ├── 02_build_splits.py
│   ├── 03_train_biencoder.py
│   ├── 04_embed_and_index.py
│   ├── 05_mine_negatives.py
│   ├── 06_train_reranker.py
│   ├── 07_eval_retrieval.py
│   ├── 08_generate_answers.py
│   └── 09_eval_faithfulness.py
├── notebooks/
│   ├── 01_eda.ipynb               # Phase 1 outputs → paper Section 2
│   └── 02_results.ipynb           # Phase 7/9 outputs → paper Section 5
└── artifacts/                     # gitignored, see 0.2
```

**Rule for agents:** all logic lives in `src/`. Scripts parse arguments, load config, call one function, write one artifact. Notebooks import from `src/` and never define analysis logic inline — they exist to render figures and tables.

### 0.2 Artifact registry

```
artifacts/
├── raw/                    # untouched HF download
├── interim/                # filtered/audited data + audit reports
├── processed/              # final train/val/test jsonl
├── models/
│   ├── biencoder/{run_id}/
│   └── reranker/{run_id}/
├── indexes/{config_name}/{tier}/
├── runs/{run_id}/          # metrics.json, config snapshot, git sha
└── figures/
```

Every phase writes a `manifest.json` alongside its output containing: input artifact hashes, config snapshot, git SHA, timestamp, random seed. A phase that cannot reproduce its inputs from the manifest is not done.

### 0.3 GPU allocation

| Workload | Device | Note |
|---|---|---|
| Bi-encoder training | 48 GB | GradCache lets effective batch exceed physical |
| Reranker training | 48 GB | Cross-encoder is heavier per pair |
| Corpus embedding | 12 GB | Runs in parallel with training |
| BM25 | CPU | — |
| gemma4 generation | 48 GB (offline) | Never contends with training; sequence the phases |
| NLI judge | 12 GB | DeBERTa-large is small |

Set `CUDA_VISIBLE_DEVICES` per script; never assume device 0.

### 0.4 Determinism

Seed `random`, `numpy`, `torch`, and set `torch.use_deterministic_algorithms(True)` where it does not tank throughput. Log the seed in every manifest. Sweeps vary the seed deliberately and record it.

### 0.5 Environment

Python 3.11, CUDA 12.x. Pin exact versions in `requirements.txt` at first successful run and never float them again. Core: `sentence-transformers`, `transformers`, `datasets`, `torch`, `pymilvus` (or `faiss-gpu`), `rank_bm25`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`.

---

## Phase 0 — Data Acquisition

**Purpose:** pull source data once, immutably.

**Script:** `scripts/00_download.py`
**Inputs:** none
**Outputs:** `artifacts/raw/medembed/`, `artifacts/raw/nfcorpus/`

```python
# src/lyra_capstone/data/load.py

def download_medembed(dest: Path) -> Path:
    """Pull abhinand/MedEmbed-training-triplets-v1 from HF.

    Fetch all three configs: 'merged' (21,689 rows, all negatives per
    query), 'corpus' (27,590 passages), 'queries' (21,689).
    Persist as parquet. Do NOT modify.

    Returns path to the download directory.
    """
    # TODO(agent): use datasets.load_dataset with the config names above
    # TODO(agent): assert row counts match the spec; fail loudly if not —
    #   an upstream revision change must not pass silently
    ...

def download_nfcorpus(dest: Path) -> Path:
    """Pull NFCorpus from BEIR. Expect 3,633 docs / 323 test queries.

    Includes qrels with graded relevance (0-2). Preserve grades — nDCG
    needs them; do not binarize here.
    """
    # TODO(agent): beir.datasets loader or direct HF mirror
    ...
```

**Acceptance criteria**
- [ ] MedEmbed row counts assert to 232,684 / 27,590 / 21,689
- [ ] NFCorpus asserts to 3,633 docs, 323 test queries
- [ ] `artifacts/raw/` is read-only after this phase (`chmod -R a-w`)
- [ ] Manifest records the HF dataset revision SHA, not just the name

**Runtime:** minutes. **Blocker risk:** low.

---

## Phase 1 — Audit and Filter

**Purpose:** produce the filtered dataset *and* the EDA evidence for paper Section 2. This phase is on the critical path — its output determines dataset size, which determines batch sizing in Phase 3.

**Script:** `scripts/01_audit_filter.py`
**Inputs:** `artifacts/raw/medembed/`
**Outputs:** `artifacts/interim/filtered.parquet`, `artifacts/interim/audit_report.json`, `artifacts/figures/eda_*.png`

### 1.1 Anaphoric query filter (D5 — load-bearing)

```python
# src/lyra_capstone/data/filters.py

@dataclass
class AnaphoricFilterResult:
    is_anaphoric: bool
    matched_pattern: str | None
    clinical_terms_found: list[str]

def is_anaphoric_query(query: str, clinical_vocab: set[str]) -> AnaphoricFilterResult:
    """Flag context-dependent queries with no unique correct answer.

    Positive examples (drop these):
        "What was the outcome of the patient's treatment?"
        "What was the patient's condition at birth?"
        "How did the patient fare after the surgery?"

    Negative examples (keep these):
        "thyroid nodule symptoms"
        "What NIHSS score indicates severe stroke?"
        "TEVAR postoperative complications"

    Rule: flag if the query contains a bare definite reference to the
    patient (the patient / the patient's / his / her + generic noun)
    AND contains no term from `clinical_vocab`.

    The second clause is what saves "What was the patient's NIHSS score?"
    — anaphoric in form, but the clinical entity makes it retrievable.
    """
    # TODO(agent): compile the definite-reference patterns as regex
    # TODO(agent): clinical_vocab from UMLS/MeSH term dump OR scispacy NER;
    #   choose on install friction, not marginal accuracy — see Open Items
    ...

def validate_anaphoric_filter(
    df: pd.DataFrame, n_sample: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Emit a stratified sample for hand-labeling.

    50 flagged + 50 unflagged, shuffled, with a blank `human_label`
    column. The team fills this in manually. Filter precision and
    recall computed from it go in the paper — an unvalidated filter
    that removes a third of the dataset will not survive review.
    """
    # TODO(agent): write to artifacts/interim/filter_validation_sample.csv
    ...
```

### 1.2 False-negative detection

```python
def score_negative_plausibility(
    df: pd.DataFrame, model_name: str = "BAAI/bge-m3", device: str = "cuda:1"
) -> pd.DataFrame:
    """Score every (query, negative) pair with the STOCK encoder.

    Stock, not fine-tuned — this runs before training and must not
    depend on it.

    Adds columns: neg_similarity, neg_reuse_count (how many distinct
    queries share this neg_id). High reuse indicates generic filler
    like "The patient was provided with adequate postoperative care."
    """
    # TODO(agent): batch encode, cosine, 12GB card
    ...

def flag_false_negatives(df: pd.DataFrame, sim_threshold: float) -> pd.DataFrame:
    """Drop negatives above threshold. Threshold set from the similarity
    distribution, not picked a priori — inspect the histogram first and
    record the chosen value in config."""
    # TODO(agent)
    ...
```

### 1.3 Degenerate positive filter

```python
def filter_degenerate_positives(df: pd.DataFrame, min_tokens: int) -> pd.DataFrame:
    """Drop near-contentless positives ("Nurick scale 1.").

    Passages range 5-993 chars; the low tail is unusable as a retrieval
    target. Report the count removed.
    """
    # TODO(agent)
    ...
```

### 1.4 EDA outputs (feed `notebooks/01_eda.ipynb`)

Required figures, each destined for paper Section 2:

| Figure | Content |
|---|---|
| `eda_lengths.png` | Query and passage length distributions |
| `eda_anaphoric.png` | Anaphoric rate + filter precision/recall from hand labels |
| `eda_negatives.png` | Negative similarity distribution + reuse frequency |
| `eda_query_types.png` | Keyword-style vs. natural-question breakdown |
| `eda_lexical_overlap.png` | Query↔positive token overlap (motivates the BM25 baseline) |
| `eda_specialty.png` | Topical/specialty coverage of the corpus |

**Acceptance criteria**
- [ ] `audit_report.json` reports every removal rate with absolute counts
- [ ] Hand-validation sample written; filter precision/recall computable once labeled
- [ ] Surviving unique-query count logged prominently — **this number gates Phase 3**
- [ ] All six figures render at publication quality (300 dpi, no dark mode, readable at column width)
- [ ] If surviving queries < 8,000, stop and escalate — the filter is too aggressive

**Runtime:** ~2 hours compute, plus manual labeling. **Blocker risk: HIGH** — everything downstream waits on this.

---

## Phase 2 — Splits and Dataset Build

**Purpose:** leak-free document-level partitioning.

**Script:** `scripts/02_build_splits.py`
**Inputs:** `artifacts/interim/filtered.parquet`
**Outputs:** `artifacts/processed/{train,val,test}.jsonl`, `artifacts/processed/corpus.jsonl`, `split_manifest.json`

```python
# src/lyra_capstone/data/splits.py

def partition_corpus(
    corpus_ids: list[str], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> dict[str, set[str]]:
    """Partition the 27,590 PASSAGES — not the queries — 80/10/10."""
    # TODO(agent)
    ...

def assign_queries_to_splits(
    df: pd.DataFrame, partitions: dict[str, set[str]]
) -> pd.DataFrame:
    """Each query inherits the split of its pos_id."""
    # TODO(agent)
    ...

def sanitize_training_negatives(
    train_df: pd.DataFrame,
    partitions: dict[str, set[str]],
    corpus: dict[str, str],
    n_negatives: int,
    device: str = "cuda:1",
) -> pd.DataFrame:
    """Remove any training negative pointing into val/test, then re-mine
    replacements from the TRAIN partition only using stock bge-m3 top-k.

    This is the leakage guarantee: after this call, no test-partition
    passage has been seen during training in any role, positive or
    negative.
    """
    # TODO(agent): encode train partition once, reuse for all re-mining
    ...

def audit_leakage(train_df, val_df, test_df, partitions) -> dict:
    """Hard assertion pass. Must return zero violations across:
      - test pos_ids appearing anywhere in train
      - test/val pos_ids appearing as train negatives
      - duplicate queries straddling splits (exact + near-dup)
    Raise on any violation. Do not warn — raise.
    """
    # TODO(agent)
    ...
```

**Note on the eval index:** at evaluation time the *full* 27,590-passage corpus is indexed for every configuration so the distractor pool is identical across the results table. Only test-partition queries are scored. The split governs training exposure, not index membership.

**Acceptance criteria**
- [ ] `audit_leakage` returns zero violations and the assertion is in CI, not just a notebook
- [ ] Split sizes recorded in `split_manifest.json`
- [ ] Re-mined negatives average the configured count per query
- [ ] Re-running with the same seed reproduces splits byte-identically

**Runtime:** ~1 hour. **Blocker risk:** medium — silent leakage invalidates every downstream number.

---

## Phase 3 — Experiment A: Bi-Encoder Training

**Purpose:** domain-adapt bge-m3, plus the hyperparameter sweep that Section 4 of the paper requires.

**Script:** `scripts/03_train_biencoder.py`
**Inputs:** `artifacts/processed/`
**Outputs:** `artifacts/models/biencoder/{run_id}/`, `artifacts/runs/{run_id}/metrics.json`

```python
# src/lyra_capstone/models/biencoder.py

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

def train_biencoder(cfg: BiEncoderConfig, train_path: Path, val_path: Path, out_dir: Path) -> Path:
    """Fine-tune bge-m3 with CachedMultipleNegativesRankingLoss.

    GradCache is not optional. In-batch negatives are the dominant
    training signal for contrastive retrieval, and the cached loss is
    what buys an effective batch of 256+ on a 48 GB card at this model
    scale (~568M params, XLM-R-large).

    Evaluate nDCG@10 on val every N steps; early-stop on it. Save the
    best checkpoint, not the last.
    """
    # TODO(agent): sentence_transformers.losses.CachedMultipleNegativesRankingLoss
    # TODO(agent): InformationRetrievalEvaluator on the val split
    # TODO(agent): persist loss curve to artifacts/figures/ — paper Section 5
    #   explicitly asks for training curves
    ...

def run_sweep(base_cfg: BiEncoderConfig, grid: dict[str, list], out_dir: Path) -> pd.DataFrame:
    """Grid over learning_rate x epochs x n_hard_negatives.

    An epoch is a few hundred steps at this dataset size — minutes, not
    hours. There is no excuse for a single run, and the template
    requires comparative optimization content.

    Returns a dataframe of (config, val_ndcg@10) for the results section.
    """
    # TODO(agent): suggested grid
    #   learning_rate:    [1e-5, 2e-5]
    #   epochs:           [1, 2, 3]
    #   n_hard_negatives: [4, 8]
    ...
```

**Acceptance criteria**
- [ ] Training curve saved and shows convergence, not divergence
- [ ] Best checkpoint selected by val nDCG@10, never by final-epoch loss
- [ ] Sweep table written for the paper
- [ ] Output embedding dimension asserts to 1024 by probing a test embedding, not by hardcoding
- [ ] Peak VRAM logged — needed for the reproducibility appendix

**Runtime:** minutes per run, ~2 hours for the full sweep. **Blocker risk:** low.

---

## Phase 4 — Corpus Embedding and Index Construction

**Purpose:** build every index the config matrix needs.

**Script:** `scripts/04_embed_and_index.py`
**Outputs:** `artifacts/indexes/{config_name}/{tier}/`

Six retrievers × two tiers. Build all of them here so Phase 7 is pure measurement.

| Config | Retriever | Index type |
|---|---|---|
| `bm25` | BM25 | Lexical, CPU |
| `stock_bgem3` | Stock bge-m3 | Dense 1024 COSINE |
| `medembed` | MedEmbed-large-v0.1 | Dense 1024 COSINE |
| `ft_bgem3` | Fine-tuned bge-m3 | Dense 1024 COSINE |

Configs 5 and 6 reuse `ft_bgem3` — reranking is a post-retrieval stage, not a separate index.

```python
# src/lyra_capstone/indexes/dense.py

def build_dense_index(model_name_or_path: str, corpus: dict[str, str], out_dir: Path, device: str = "cuda:1") -> Path:
    """Embed corpus and persist a searchable index.

    Probe the dimension from a test embedding and assert against the
    collection schema. Never hardcode 1024 — the LLD's standing rule.

    Tier 1 corpus: 27,590 passages. Tier 2 (NFCorpus): 3,633.
    Both are small; use Milvus for LLD parity or FAISS for simplicity.
    """
    # TODO(agent): decide Milvus vs FAISS ONCE and record it — see Open Items
    ...

# src/lyra_capstone/indexes/lexical.py

def build_bm25_index(corpus: dict[str, str], out_dir: Path) -> Path:
    """rank_bm25 over the same corpus.

    Expect this to be competitive with — possibly better than —
    stock bge-m3 on clinical notes. Exact drug names, abbreviations,
    and numeric scores favor lexical matching. That is a finding,
    not a bug; do not tune it away.
    """
    # TODO(agent): document the tokenization choice; it materially
    #   affects BM25 on clinical text
    ...
```

**Acceptance criteria**
- [ ] Eight indexes exist (4 retrievers × 2 tiers)
- [ ] Dimension probe asserts 1024 for all three dense models
- [ ] Tier 1 index contains all 27,590 passages, not just the test partition
- [ ] Index build time and size logged

**Runtime:** ~1 hour. **Blocker risk:** low.

---

## Phase 5 — Hard-Negative Mining for the Reranker

**Purpose:** build reranker training data from the distribution it will actually see.

**Script:** `scripts/05_mine_negatives.py`
**Inputs:** `artifacts/models/biencoder/{best_run}/`, `artifacts/processed/train.jsonl`
**Outputs:** `artifacts/processed/reranker_train.jsonl`

```python
# src/lyra_capstone/mining/hard_negatives.py

def mine_reranker_negatives(
    retriever_path: Path,
    train_df: pd.DataFrame,
    corpus: dict[str, str],
    train_partition: set[str],
    top_n: int = 50,
    n_negatives: int = 8,
) -> pd.DataFrame:
    """Retrieve top-N with the FINE-TUNED retriever; non-positives become
    negatives.

    Two rules that are easy to get wrong:
      1. Use the fine-tuned retriever, not stock. The reranker's entire
         job is to fix the fine-tuned retriever's ranking errors, so it
         must train on those errors.
      2. Restrict candidates to the TRAIN partition. Mining over the
         full corpus reintroduces the leakage Phase 2 eliminated.

    Do NOT reuse the original MedEmbed negatives here.
    """
    # TODO(agent)
    ...
```

**Acceptance criteria**
- [ ] Zero mined negatives fall outside the train partition (assert)
- [ ] Negative difficulty distribution plotted — should skew harder than Phase 2's negatives; if not, the mining is broken
- [ ] ~8 negatives per positive on average

**Runtime:** ~1 hour. **Blocker risk:** medium — the two rules above are the common failure mode.

---

## Phase 6 — Experiment B: Reranker Training

**Script:** `scripts/06_train_reranker.py`
**Outputs:** `artifacts/models/reranker/{run_id}/`

```python
# src/lyra_capstone/models/reranker.py

@dataclass
class RerankerConfig:
    base_model: str = "BAAI/bge-reranker-v2-m3"
    max_seq_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 2
    n_negatives_per_positive: int = 8
    eval_steps: int = 200          # deliberately frequent — see below
    seed: int = 42

def train_reranker(cfg: RerankerConfig, train_path: Path, val_path: Path, out_dir: Path) -> Path:
    """Train a cross-encoder with binary cross-entropy on relevance.

    Cross-encoders converge fast and overfit faster. Evaluate val
    nDCG@10 frequently and stop the moment it turns over, even
    mid-epoch. A checkpoint from step 600 beating one from step 2000
    is the expected outcome, not a red flag.
    """
    # TODO(agent): sentence_transformers.cross_encoder.CrossEncoder
    # TODO(agent): checkpoint on every eval; keep best by val nDCG@10
    ...

def rerank(model_path: Path, query: str, candidates: list[str], top_k: int) -> list[int]:
    """Re-score and return reordered candidate indices.

    Called at N=30-50 candidates. Latency belongs to the chat GPU
    lease per LLD 5.1 — measure and record it, it goes in the paper.
    """
    # TODO(agent)
    ...
```

**Acceptance criteria**
- [ ] Val nDCG@10 curve shows the turnover point; early stopping documented
- [ ] Per-query rerank latency measured at N=30 and N=50
- [ ] Best checkpoint beats the stock reranker on val, or the negative result is documented explicitly

**Runtime:** ~2 hours. **Blocker risk:** low.

---

## Phase 7 — Retrieval Evaluation Matrix

**Purpose:** the six-row, two-tier results table. This is the paper's spine.

**Script:** `scripts/07_eval_retrieval.py`
**Outputs:** `artifacts/runs/eval_retrieval/results.json`, `results.csv`

```python
# src/lyra_capstone/eval/metrics.py

def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float: ...
def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float: ...
def ndcg_at_k(ranked_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    """NFCorpus qrels are graded 0-2. Use the grades; do not binarize."""
    ...

# src/lyra_capstone/eval/significance.py

def paired_bootstrap(
    scores_a: np.ndarray, scores_b: np.ndarray, n_resamples: int = 10_000, seed: int = 42
) -> dict:
    """Per-query paired bootstrap between adjacent configs.

    Returns mean difference, 95% CI, p-value. Cheap to run and
    preempts "is that two-point gain real?"
    """
    # TODO(agent)
    ...

# src/lyra_capstone/eval/retrieval.py

def run_config_matrix(cfg: EvalConfig) -> pd.DataFrame:
    """Evaluate all six configs on both tiers.

    Configs:
      1. bm25                              (lexical reference)
      2. stock_bgem3                       (primary baseline)
      3. medembed_large                    (prior art, same training data)
      4. ft_bgem3                          (Experiment A)
      5. ft_bgem3 + stock reranker         (isolates "reranking helps")
      6. ft_bgem3 + ft reranker            (Experiment B, full system)

    Tiers:
      tier1 — held-out MedEmbed clinical test split, 27,590-doc index
      tier2 — NFCorpus, 323 test queries, 3,633-doc index

    Config 5 is the ablation that justifies Experiment B existing.
    Retrieve N=50 before reranking; report metrics at k in {1,5,10}.
    """
    # TODO(agent): persist PER-QUERY scores, not just aggregates —
    #   significance testing and error analysis both need them
    ...
```

**Acceptance criteria**
- [ ] 12 cells populated (6 configs × 2 tiers), each with Recall@{1,5,10}, MRR, nDCG@10
- [ ] Per-query scores persisted
- [ ] Paired bootstrap run between configs 2→4, 4→5, 5→6
- [ ] Results table renders in `notebooks/02_results.ipynb` with best-per-metric bolded

**Interpretation guardrails — decide these now, before seeing numbers:**
- BM25 beating stock bge-m3 is expected and reportable.
- The fine-tune landing near MedEmbed-large validates the pipeline; it is not a competition.
- Tier 2 gains being smaller than Tier 1 is the honest transfer story, not a failure. Report the gap.

**Runtime:** ~3 hours. **Blocker risk:** low.

---

## Phase 8 — Answer Generation

**Purpose:** produce answers for faithfulness scoring, with the generator held rigidly constant.

**Script:** `scripts/08_generate_answers.py`
**Outputs:** `artifacts/runs/generation/{config_name}_{tier}.jsonl`

```python
# src/lyra_capstone/generation/generate.py

def generate_answers(
    config_name: str,
    retrieved: dict[str, list[str]],
    prompt_template: str,
    n_queries: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate with gemma4, frozen.

    The experiment is only valid if the ONLY thing differing across
    configs is retrieved context. Therefore:
      - identical prompt template everywhere
      - identical decoding parameters (temperature, top_p, max_tokens)
      - identical query subset across all configs (same seed, same 400)
      - identical top-k context passages passed in

    Log the exact prompt and decoding params to the manifest.
    """
    # TODO(agent): sample the query subset ONCE, persist the id list,
    #   reuse for every config — resampling per config invalidates
    #   the paired comparison
    ...
```

**Acceptance criteria**
- [ ] Same query IDs across all six configs (assert on set equality)
- [ ] Prompt template and decoding params byte-identical across runs
- [ ] 300–500 queries per config per tier
- [ ] Raw generations persisted for error analysis

**Runtime:** ~4 hours. Sequence after training — gemma4 wants the 48 GB card. **Blocker risk:** low.

---

## Phase 9 — Faithfulness Evaluation

**Purpose:** the contribution. Does better retrieval yield more trustworthy answers with the generator unchanged?

**Script:** `scripts/09_eval_faithfulness.py`
**Outputs:** `artifacts/runs/faithfulness/results.json`

```python
# src/lyra_capstone/eval/faithfulness.py

def split_into_claims(answer: str) -> list[str]:
    """Sentence-split the answer into atomic claims."""
    # TODO(agent)
    ...

def score_claim_entailment(
    claim: str, context: str, nli_model, threshold: float, max_premise_tokens: int = 512
) -> float:
    """Entailment of one claim against retrieved context.

    NLI models cap at 512 tokens and the context is longer. Chunk the
    premise and take MAX entailment across chunks — a claim supported
    by any part of the context is grounded.
    """
    # TODO(agent): DeBERTa-v3-large-MNLI class model on the 12GB card
    ...

def validate_judge(sample_path: Path) -> dict:
    """Judge validation against ~50 hand-labeled claim/context pairs.

    The entailment threshold is a free parameter. Report agreement
    (Cohen's kappa) with human labels and justify the chosen value.
    An unvalidated judge cannot carry the headline claim — a reviewer
    will go straight here.
    """
    # TODO(agent): emit the labeling sheet, then compute agreement
    ...

def compute_faithfulness_suite(generations: pd.DataFrame, nli_model) -> dict:
    """Four metrics. Never report faithfulness alone.

      faithfulness    — fraction of claims entailed by retrieved context
      answer_relevance— does the answer address the question
      context_precision— fraction of retrieved context that was useful
      abstention_rate — fraction of "the context does not contain this"

    ABSTENTION IS THE CONFOUND. A model that refuses to answer is
    trivially 100% faithful. Two readings:

      abstention DOWN + faithfulness UP  -> clean causal story, the
                                            claim holds
      abstention UP  + faithfulness UP   -> null result wearing a
                                            disguise; report it as such

    Track abstention from the first run. Discovering this in week 8
    is a project-level failure.
    """
    # TODO(agent): abstention detection via refusal-phrase matching
    #   PLUS a length heuristic; validate on a sample
    ...
```

**Acceptance criteria**
- [ ] All four metrics reported per config per tier — no exceptions
- [ ] Judge validated against hand labels; kappa reported
- [ ] Abstention analyzed explicitly in the write-up, whichever direction it moves
- [ ] Faithfulness deltas significance-tested with the same paired bootstrap

**Runtime:** ~3 hours plus manual labeling. **Blocker risk: HIGH** — this carries the thesis claim.

---

## Phase 10 — Analysis and Paper Assets

**Script:** none — `notebooks/02_results.ipynb`
**Outputs:** every figure and table the paper needs

### Notebook cell plan

| Cell | Content |
|---|---|
| 1 | Imports, load all run artifacts |
| 2 | Table 1 — six-config × two-tier retrieval results, best bolded |
| 3 | Figure — training curves for both models |
| 4 | Table 2 — faithfulness suite, four metrics per config |
| 5 | Figure — retrieval gain vs. faithfulness gain scatter (**the money plot** for the central claim) |
| 6 | Table 3 — significance test results between adjacent configs |
| 7 | Error analysis — queries where fine-tuned wins/loses vs. baseline, with clinical-abbreviation examples |
| 8 | Optional: catastrophic-forgetting check on a non-medical BEIR slice |
| 9 | Export all figures at 300 dpi to `artifacts/figures/` |

**Error analysis focus:** pull concrete cases where the fine-tune fixes an abbreviation failure (TEVAR, NIHSS, Clavien-Dindo). Those examples are what make the domain-adaptation mechanism legible to a semi-technical reader, and the paper's Section 5 needs them.

**Acceptance criteria**
- [ ] Every figure at 300 dpi, no dark-mode or console screenshots
- [ ] Every table has a caption and an in-text reference planned
- [ ] The scatter in cell 5 either supports or refutes the hypothesis unambiguously

---

## 11. Critical Path and Parallelization

```
Phase 0 ──> Phase 1 ══> Phase 2 ══> Phase 3 ──> Phase 4 ──> Phase 5 ──> Phase 6
                                                    │                      │
                                                    └──────────┬───────────┘
                                                               ▼
                                                          Phase 7 ──> Phase 8 ──> Phase 9 ──> Phase 10

══ critical path, blocking
```

**Parallelizable:**
- BM25 and MedEmbed index builds (Phase 4) can run during Phase 3 training — different devices
- Phase 1 hand-labeling proceeds while automated filtering runs
- Phase 9 judge validation labeling proceeds during Phase 8 generation
- `notebooks/01_eda.ipynb` and the paper's Section 2 can be written the moment Phase 1 lands

**Three-person split suggestion:** one owns Phases 1–2 and paper Section 2; one owns Phases 3–6 and Section 4; one owns Phases 7–10 and Section 5. The Phase 2 leakage audit should be reviewed by someone who did not write it.

---

## 12. Open Items — Resolve Before Coding

| # | Item | Blocks | Note |
|---|---|---|---|
| 1 | Milvus vs. FAISS for dense indexes | Phase 4 | Milvus gives LLD parity; FAISS is faster to stand up. Pick once, record it. |
| 2 | UMLS/MeSH term list vs. scispacy NER for clinical vocab | Phase 1 | Choose on install friction. Accuracy difference is marginal at this filter's job. |
| 3 | Exact NLI judge checkpoint and its license | Phase 9 | Confirm before building the harness around it |
| 4 | Negative-similarity threshold for false-positive flagging | Phase 1 | Set from the histogram, not a priori |
| 5 | Whether the forgetting check makes the paper or an appendix | Phase 10 | Low cost either way |

---

## 13. Definition of Done

- [ ] All ten phases pass their acceptance criteria
- [ ] `audit_leakage` returns zero violations
- [ ] Six-config × two-tier retrieval table complete with significance tests
- [ ] Faithfulness suite complete with a validated judge and abstention analysis
- [ ] Every number in the paper traceable to an artifact manifest
- [ ] Paper Section 2 edits from `capstone_implementation_spec_v2.md` §9 applied
- [ ] A fresh clone plus `requirements.txt` reproduces Phase 1 output byte-identically