# Amazon ML Challenge 2026: Business Entity Resolution

Scalable, high-precision machine learning pipeline for large-scale multi-country Business Entity Resolution (BER). Developed for the Amazon ML Challenge 2026 on Unstop.

The system resolves 1,732,544 canonical reference entities (Source 1) against ~10 million noisy, unstructured candidate records (Source 2 and Source 3) across three distinct national jurisdictions: India, the United States, and France.

---

## 1. Problem Overview & Evaluation Metric

Entity Resolution under this challenge presents extreme scale, synthetic noise corruptions, and an asymmetric evaluation metric:

- **Data Volume**: 1.73M reference records, ~10M candidate pool records.
- **Noise Dimensions**: Synthetic character transpositions, domain conversions, leet-speak substitutions, clause permutations, multi-lingual Indic scripts (Devanagari, Tamil, Kannada, Bengali), and OCR noise.
- **Evaluation Metric**: Macro-averaged F0.5 score:

$$F_{0.5} = (1 + 0.5^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = 1.25 \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

### Critical Metric Invariants
1. **Asymmetric Precision Penalty**: False positives are penalized 4x to 5x heavier than false negatives. A single false match on a singleton entity (an entity with 0 true matches) drops its score from 1.0 directly to 0.0.
2. **Candidate Uniqueness Invariant**: Empirical analysis of 7,638,365 training ground-truth pairs reveals that candidate entities belong to at most one reference entity (0.0000% multi-assignment). Any valid pipeline must enforce a strict 1-to-1 maximum-weight global arbitration.

---

## 2. System Architecture

The pipeline consists of four sequential, highly optimized stages:

```text
[Source 1, 2, 3 Data]
         │
         ▼
[Stage 1: Two-Tier Hybrid Dual-Pool Blocker]
  ├── Exact Continuous Compact Brand Index (+30.0)
  ├── Postal Code + House Number Super-Anchor (+35.0)
  ├── Compound Street Number & Prefix Anchor (+25.0)
  └── Two-Tier Frequency-Capped Brand Token Index (len <= 2500)
         │
         ▼
[Stage 2: Ultra-Low Memory Pre-Parsing (__slots__)]
  └── Compact C-level struct representations (~120 bytes/record, zero disk paging)
         │
         ▼
[Stage 3: Vectorized Feature Extraction & LightGBM Inference]
  ├── 28-Dimensional Pairwise Feature Extractor
  └── Multi-Threaded LightGBM v4 Scoring (Threshold >= 0.50)
         │
         ▼
[Stage 4: Global Maximum-Weight Arbitration]
  └── Greedy 1-to-1 candidate bipartite matching & Singleton Protection
         │
         ▼
[Outputs: matching_results.tsv & candidate_pairs.tsv]
```

### Stage 1: Two-Tier Hybrid Dual-Pool Blocker (`CandidateGenerator`)
Standard inverted indices suffer from quadratic query explosion on high-frequency tokens (e.g., `kumar`, `traders`, `enterprises` in India). The Two-Tier Blocker resolves this by:
- Capping informative token indices to `len(m) <= 2500` with logarithmic IDF weighting.
- Providing deterministic fallbacks: if an entity yields 0 candidates from addresses or rare words, it samples the top 25 candidates from frequent tokens.
- Running continuous compact brand string lookups (removing punctuation and whitespace) to capture domain-name corruptions and spacing mutations.
- Indexing building numbers combined with postal codes and street words for address matching.

### Stage 2: Memory-Compact Record Architecture (`FastParsedRecord`)
Using Python `__slots__` eliminates instance `__dict__` overhead, shrinking memory consumption from ~1,080 bytes to ~120 bytes per record. The entire candidate pool of 4.7 million records fits within ~750 MB of RAM, completely avoiding OS disk swapping and pagefile thrashing.

### Stage 3: 28-Dimensional Feature Extraction & LightGBM Scoring
Candidate pairs are scored using a 28-dimensional feature vector:
1. **Brand Similarities**: RapidFuzz token sort ratio, token set ratio, standard ratio, partial ratio, and Jaro-Winkler distance.
2. **Compact Brand Analysis**: Exact compact match, containment ratio, length coverage, and brand token Jaccard.
3. **Address Similarities**: Address token sort ratio, token set ratio, address ratio, length ratio, and address token Jaccard.
4. **Physical Anchors**: Building number exact match, building number conflict, postal code exact match, phone number match, and postal code substring match.
5. **Jurisdiction Features**: 3-digit postal sorting district prefix match, French 2-digit department match, state exact match, and state conflict penalty.
6. **Robustness Signals**: Normalized Levenshtein distance, generic brand type flag, and Indic non-ASCII script detection.

The LightGBM classifier (`models/lgbm_v4.txt`) was trained on 3,037,899 candidate pairs incorporating 2,868,163 real hard-negative distractors (1:16.9 ratio). Because the model is calibrated against massive real-world distractors, a decision threshold of `0.50` maximizes Macro F0.5.

### Stage 4: Global Maximum-Weight Arbitration
To enforce candidate uniqueness and prevent false positives on singletons, the arbitration stage maps each candidate entity to its single highest-probability Source 1 reference entity. Candidates scoring below the decision threshold or conflicting with higher-confidence matches are discarded.

---

## 3. Directory Layout

```text
.
├── dataset/
│   ├── sample_train/               # 1,000-row sample training tables
│   ├── sample_test/                # 1,000-row sample test tables
│   └── README.md                   # Dataset layout and schema documentation
├── models/
│   ├── config.json                 # Model feature configuration
│   └── lgbm_v4.txt                 # Pre-trained 28-feature LightGBM model
├── results/
│   ├── 1st_try/                    # Baseline submission metadata
│   └── 3rd_try/                    # Validated production submission outputs
├── src/
│   ├── blocking.py                 # Candidate indexing and retrieval engine
│   ├── features.py                 # 28-dimensional pairwise feature extractor
│   ├── matcher.py                  # Pairwise similarity and arbitration logic
│   ├── metrics.py                  # Exact Macro-averaged F0.5 evaluator
│   ├── preprocessor.py             # String normalization and suffix stripping
│   ├── run_pipeline.py             # Full production pipeline runner
│   ├── synthetic_noise_inverter.py # Reverse-engineered noise rules and dictionaries
│   └── indic_exact_dictionary.json # State transliteration mappings
├── utils/
│   └── validate_submission.py      # Official competition submission validator
├── package_submission.py           # Submission bundler utility
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 4. Getting Started

### Prerequisites
- Python 3.10+ (tested with Python 3.13)
- 16 GB RAM recommended for full 1.73M entity test set; 4 GB RAM sufficient for sample data.

### Installation
Clone the repository and install required packages:

```bash
git clone https://github.com/Chiranth-Janardhan-moger/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026
pip install -r requirements.txt
```

---

## 5. Running the Pipeline

### Testing on Sample Data
To verify the full pipeline end-to-end using the bundled sample dataset:

```bash
python src/run_pipeline.py \
    --data-dir dataset/sample_test \
    --output-dir results/sample_run \
    --prefix test \
    --max-candidates 65 \
    --threshold 0.50
```

### Full Test Set Inference
To execute inference across all 1,732,544 test entities:

```bash
python src/run_pipeline.py \
    --data-dir /path/to/dataset/test \
    --output-dir results/3rd_try \
    --prefix test \
    --max-candidates 65 \
    --threshold 0.50
```

Output files generated:
- `results/3rd_try/matching_results.tsv`: Source 1 entity IDs mapped to resolved matching candidate IDs.
- `results/3rd_try/candidate_pairs.tsv`: Source 1 entity IDs mapped to blocking candidate sets.

### Validating Outputs
Verify compliance against the competition formatting rules:

```bash
python utils/validate_submission.py \
    --matching results/3rd_try/matching_results.tsv \
    --candidate results/3rd_try/candidate_pairs.tsv \
    --test-dir dataset/sample_test
```

### Packaging for Submission
Create the submission archive:

```bash
python package_submission.py \
    --target-dir results/3rd_try \
    --zip-name submission_v3.zip
```

---

## 6. Official Leaderboard Progression

| Iteration | Pipeline Configuration | Official Unstop Score (Macro F0.5) | Status |
| :--- | :--- | :---: | :--- |
| **Submission 1** | Baseline String Similarity (Threshold 0.70/0.55) | `0.562` | Evaluated |
| **Submission 2** | Tuned Rules + Partial Inverted Index | `0.722` | Evaluated |
| **Submission 3** | Distractor-Hardened LightGBM + Two-Tier Blocker + Global Arbitration | `0.758` | Verified Peak |

---

## 7. License

This project is licensed under the MIT License.
