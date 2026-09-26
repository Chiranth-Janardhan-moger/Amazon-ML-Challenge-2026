# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Team Name]  
**Team Members:** Chiranth Moger and Team  
**Submission Date:** 25th September 2026  

---

## 1. Executive Summary

We present a high-throughput, precision-calibrated two-stage Business Entity Resolution architecture engineered specifically for the macro-averaged F_0.5 evaluation metric across multilingual e-commerce registries. The solution combines strict country-partitioned multi-pass inverted index blocking (achieving 99.97% recall ceiling on ground truth while pruning the comparison space by over 500,000x) with a C++ accelerated RapidFuzz pairwise matcher incorporating Unicode NFKD de-diacritization, legal entity normalization, domain name decomposition, and street address number invariants.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12.5 million records identified four primary structural noise patterns:
1. **Legal Suffix Inconsistencies:** Businesses vary arbitrarily between legal abbreviations (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, `SASU`, `EURL`, `SCI`) and trade names.
2. **DBA / Transposition Prefixes:** Substantial presence of DBAs and former entity indicators (e.g., `Zephgildecto f/k/a Sai Construction` matching `Sai Construction`, or `Mirapyra formerly Painters Local Union 634` matching `Painters Local Union 634`).
3. **Domain Name Concatenation:** E-commerce entries frequently replace spaces with continuous alphanumeric domains (e.g., `maurewilliamscolombier.com` matching `Maure Williams Colombier Inc`).
4. **Scrambled Address Tokens & Invariant Numbers:** Address components are heavily permuted across records (e.g., city/state appearing before street names), but physical street and building numbers (`6207`, `13834`, `125`) remain invariant across genuine matches.
5. **Zero Cross-Country Leakage:** Empirical verification over 345,997 ground truth matches revealed exactly 0.0000% cross-border entity matches, confirming that entity resolution partitions strictly by country.

### 2.2 Solution Strategy

**Approach Type:** Country-Partitioned Multi-Pass Inverted Index Blocking + Precision-Calibrated Pairwise Matcher.  
**Core Innovation:** 
* **Zero-Copy Columnar Partitioning:** Streaming data country-by-country (France: 259k, US: 663k, India: 810k) eliminates O(N^2) memory consumption and completes end-to-end inference across 11.7 million records in minutes.
* **Loss-Aligned Precision Thresholding:** In the macro-averaged F_0.5 metric, false merges are penalised 5x more heavily than missed matches, and singletons award a full 1.0 score only on empty predictions. The decision boundary enforces strict brand and address agreement to eliminate false positives.

---

## 3. Candidate Generation (Blocking)

To prune the 17.2 trillion pairwise space ($1.73M \times 9.97M$) into a tractable candidate pool:
* **Partitioning Key:** Exact string matching on `country`.
* **Pass 1 (Normalized Brand Token Inverted Index):** Inverted index on significant alphanumeric brand tokens of length >= 2, excluding universal stop words and legal forms. Common tokens with frequency > 2,500 are suppressed to prevent non-discriminative candidate explosion.
* **Pass 2 (Address Number Inverted Index):** Indexing on street and building numbers with leading zeros stripped (`01604` -> `1604`).
* **Pass 3 (Compact Name Index):** Inverted indexing on continuous alphanumeric strings of length >= 4 to capture concatenated domain names (`truefactory.com` -> `truefactory`).
* **Candidate Set Size:** Capped at the top 35 candidates per Source 1 entity sorted by candidate overlap score.
* **Recall Verification:** Verified empirically on 69,144 ground truth training pairs, achieving a **99.97% recall ceiling** (69,125 / 69,144 matches retained).

---

## 4. Matching Model

**Features Used:**
1. **Brand Name Similarity:**
   * Token-set ratio, token-sort ratio, and Levenshtein ratio on normalized brand tokens.
   * Alphanumeric compact string containment for domain name recognition.
2. **Address Similarity:**
   * Address token-set ratio to handle arbitrary component reordering (street, landmark, city, state).
3. **Numerical Invariant Agreement:**
   * Exact set intersection of normalized house/building numbers. Conflicting street numbers on identical street names are actively penalized to prevent false merges of adjacent addresses.

**Model Type:** Weighted Composite Metric Classifier + Precision Thresholding.  
**Threshold Selection Method:** Grid-searched over out-of-fold ground truth validation splits to maximize the macro-averaged F_0.5 metric. The optimal threshold requires Brand Similarity >= 0.70, Address Similarity >= 0.55, and non-conflicting street numbers.

---

## 5. Results & Error Analysis

* **F_0.5 Score (macro validation):** **0.8655** (conservative baseline) to **0.9082** depending on address noise calibration.
* **Common False Positives (prevented):**
  * Disjoint businesses sharing generic legal suffixes or family trade terms (e.g. `Thermal & Fils SASU` vs `Personnel & Fils SASU`, correctly rejected due to low brand similarity of 0.38).
  * Medical centers and commercial complexes sharing generic descriptors (e.g. `Elephant Centre EURL` vs `Centre Reine`, correctly rejected).
* **Common False Negatives:**
  * Rare transliterations between Latin characters and Indian regional scripts (Hindi / Kannada) where neither Latin brand tokens nor numerical components were present.

---

## 6. Conclusion

The developed pipeline resolves large-scale commercial business identities with high throughput and robust precision. By anchoring the candidate generation on physical invariants and aligning decision thresholds with the asymmetric F_0.5 metric, the solution scales efficiently to millions of records while generalizing seamlessly to unseen geographic jurisdictions like France.

---

## Appendix

### A. Code Artefacts
The full runnable pipeline is packaged under `code/business_entity_resolution/`:
* `src/preprocessor.py`: Text normalization and token extraction.
* `src/blocking.py`: Multi-pass country-partitioned candidate generator.
* `src/matcher.py`: Pairwise similarity scoring and precision thresholding.
* `src/metrics.py`: Official macro-averaged F_0.5 scorer.
* `src/run_pipeline.py`: Main entrypoint generating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
* `requirements.txt`: Pinned dependencies (`polars`, `rapidfuzz`, `scikit-learn`, `numpy`).
* `README.md`: End-to-end execution guide.
