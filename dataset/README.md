# Dataset Documentation

## Overview

This repository includes representative sample datasets (`sample_train/` and `sample_test/`) containing 1,000 records each to enable rapid testing, pipeline verification, and validation out-of-the-box without requiring gigabytes of storage.

## Data Schema

### 1. Source 1 (Reference Entities)
- File: `train_source1.tsv` / `test_source1.tsv`
- Columns:
  - `entity_id` (string): Unique identifier for the canonical business entity.
  - `business_name` (string): Normalized name of the business entity.
  - `business_address` (string): Canonical street address, locality, city, state, and postal code.
  - `country` (string): Country partition (`India`, `US`, `France`).

### 2. Source 2 & Source 3 (Noisy Candidate Pools)
- Files: `train_source2.tsv`, `train_source3.tsv` / `test_source2.tsv`, `test_source3.tsv`
- Columns:
  - `entity_id` (string): Candidate entity identifier.
  - `business_name` (string): Business name with synthetic corruptions (acronyms, leet-speak, transpositions, domain names, Indic transliterations).
  - `business_address` (string): Address record with noise (abbreviations, clause reordering, dropped fields, OCR noise).
  - `country` (string): Country partition (`India`, `US`, `France`).

### 3. Ground Truth (Training Only)
- File: `train_ground_truth.tsv`
- Columns:
  - `source1_entity_id` (string): Canonical entity ID from Source 1.
  - `matched_entity_ids` (string): Comma-separated list of true positive matching entity IDs from Source 2 and Source 3.

## Full Dataset Layout

For full training and full test evaluation:
Place the complete challenge datasets in the following directory layout:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Note: Full competition dataset files exceed 2.5 GB and are excluded from git version control.
