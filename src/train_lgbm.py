"""
Train LightGBM Pairwise Classifier for Business Entity Resolution.
Mines hard negatives directly from candidate blocking and calibrates
decision threshold directly against the Macro-averaged F_0.5 metric.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.blocking import CandidateGenerator
from src.features import extract_pairwise_features, FEATURE_NAMES
from src.metrics import macro_f_beta_score


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Load ground truth mapping from S1 ID to set of matching S2/S3 IDs."""
    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_map = {}
    for row in gt_df.iter_rows(named=True):
        sid = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if matched_str and matched_str != "nan":
            gt_map[sid] = set(matched_str.split(","))
        else:
            gt_map[sid] = set()
    return gt_map


def train_model(
    train_dir: str,
    output_model_path: str,
    output_config_path: str,
    sample_entities: int = 50000,
    max_cands: int = 25
):
    print("=" * 70)
    print("TRAINING PAIRWISE LIGHTGBM CLASSIFIER")
    print(f"Data Directory     : {train_dir}")
    print(f"Sample Entities    : {sample_entities}")
    print(f"Max Candidates     : {max_cands}")
    print(f"Model Output Path  : {output_model_path}")
    print(f"Config Output Path : {output_config_path}")
    print("=" * 70)

    t0 = time.time()

    # 1. Load Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print(f"[*] Loading ground truth from {gt_path}...")
    gt_map = load_ground_truth(gt_path)
    print(f"    Loaded ground truth for {len(gt_map):,} entities.")

    # 2. Load Source Data
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    print(f"[*] Loading Source 1 from {s1_path}...")
    s1_df = pl.read_csv(s1_path, separator="\t")
    print(f"[*] Loading Source 2 from {s2_path}...")
    s2_df = pl.read_csv(s2_path, separator="\t")
    print(f"[*] Loading Source 3 from {s3_path}...")
    s3_df = pl.read_csv(s3_path, separator="\t")

    # Sample balanced subset of entities across countries
    countries = s1_df["country"].unique().to_list()
    s1_sampled_ids = set()
    per_country_sample = sample_entities // len(countries)

    for country in countries:
        c_ids = s1_df.filter(pl.col("country") == country)["entity_id"].to_list()
        random.seed(42)
        s1_sampled_ids.update(random.sample(c_ids, min(per_country_sample, len(c_ids))))

    print(f"[*] Total sampled entities for dataset creation: {len(s1_sampled_ids):,}")

    # Train / Val entity split (80/20) with zero entity leakage
    sampled_list = sorted(list(s1_sampled_ids))
    random.seed(42)
    random.shuffle(sampled_list)
    split_idx = int(0.80 * len(sampled_list))
    train_s1_ids = set(sampled_list[:split_idx])
    val_s1_ids = set(sampled_list[split_idx:])

    print(f"    Train S1 entities: {len(train_s1_ids):,}")
    print(f"    Val S1 entities  : {len(val_s1_ids):,}")

    # 3. Mine Candidate Pairs and Extract Features
    blocking_engine = CandidateGenerator(max_candidates_per_entity=max_cands)

    X_train, y_train = [], []
    X_val, y_val = [], []
    val_candidate_pairs: Dict[str, List[Tuple[str, List[float]]]] = {sid: [] for sid in val_s1_ids}

    for country in countries:
        print(f"[*] Mining pairs for country: {country}...")
        s1_c = s1_df.filter(pl.col("country") == country)
        s2_c = s2_df.filter(pl.col("country") == country)
        s3_c = s3_df.filter(pl.col("country") == country)
        cands_c = pl.concat([s2_c, s3_c])

        # Filter s1_c to sampled entities
        s1_c_sampled = s1_c.filter(pl.col("entity_id").is_in(list(s1_sampled_ids)))
        if len(s1_c_sampled) == 0:
            continue

        cands_dict = blocking_engine.retrieve_candidates_for_country(s1_c_sampled, cands_c)

        s1_records = dict(zip(
            s1_c_sampled["entity_id"].to_list(),
            zip(s1_c_sampled["business_name"].fill_null("").to_list(), s1_c_sampled["business_address"].fill_null("").to_list())
        ))
        cand_records = dict(zip(
            cands_c["entity_id"].to_list(),
            zip(cands_c["business_name"].fill_null("").to_list(), cands_c["business_address"].fill_null("").to_list())
        ))

        for s1_id, candidate_ids in cands_dict.items():
            if s1_id not in s1_records:
                continue
            n1, a1 = s1_records[s1_id]
            true_matches = gt_map.get(s1_id, set())

            is_train = s1_id in train_s1_ids

            for cid in candidate_ids:
                if cid not in cand_records:
                    continue
                n2, a2 = cand_records[cid]
                label = 1 if cid in true_matches else 0
                feats = extract_pairwise_features(n1, a1, n2, a2, cid)

                if is_train:
                    X_train.append(feats)
                    y_train.append(label)
                else:
                    X_val.append(feats)
                    y_val.append(label)
                    val_candidate_pairs[s1_id].append((cid, feats))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    pos_train = np.sum(y_train == 1)
    neg_train = np.sum(y_train == 0)
    print(f"[+] Training Set: {len(y_train):,} pairs (Pos: {pos_train:,} [{pos_train/len(y_train):.1%}], Neg: {neg_train:,})")
    print(f"[+] Val Set     : {len(y_val):,} pairs (Pos: {np.sum(y_val==1):,}, Neg: {np.sum(y_val==0):,})")

    # 4. Train LightGBM Model
    scale_pos = max(1.0, float(neg_train) / max(1, pos_train) * 0.5)
    print(f"[*] Training LightGBM with scale_pos_weight={scale_pos:.2f}...")

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)]
    )

    # 5. Threshold Calibration for Macro-averaged F_0.5
    print("-" * 70)
    print("[*] Calibrating decision threshold against Macro-averaged F_0.5...")

    val_preds_prob = clf.predict_proba(X_val)[:, 1]

    # Pre-populate index for fast lookup
    prob_idx = 0
    val_entity_preds = {}
    for s1_id in val_s1_ids:
        cands_with_feats = val_candidate_pairs.get(s1_id, [])
        cand_probs = []
        for cid, _ in cands_with_feats:
            cand_probs.append((cid, float(val_preds_prob[prob_idx])))
            prob_idx += 1
        val_entity_preds[s1_id] = cand_probs

    val_gt_subset = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}

    best_thresh = 0.60
    best_f05 = 0.0

    thresholds_to_test = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    for thresh in thresholds_to_test:
        preds_for_thresh = {}
        for sid, cand_probs in val_entity_preds.items():
            preds_for_thresh[sid] = {cid for cid, p in cand_probs if p >= thresh}

        score = macro_f_beta_score(val_gt_subset, preds_for_thresh, beta=0.5)
        print(f"    Threshold {thresh:.2f} -> Validation Macro F_0.5: {score:.5f}")

        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    print(f"[+] Optimal Threshold: {best_thresh:.2f} (Macro F_0.5: {best_f05:.5f})")

    # 6. Save Model and Configuration
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    clf.booster_.save_model(output_model_path)
    print(f"[+] Model saved to {output_model_path}")

    config = {
        "model_type": "LightGBM",
        "best_threshold": best_thresh,
        "best_val_f05": best_f05,
        "feature_names": FEATURE_NAMES,
        "num_train_pairs": int(len(y_train)),
        "num_val_pairs": int(len(y_val)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_config_path, "w", encoding="utf-8") as f_cfg:
        json.dump(config, f_cfg, indent=2)
    print(f"[+] Config saved to {output_config_path}")
    print(f"[+] Training completed in {time.time() - t0:.2f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Pairwise Classifier")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train dataset directory")
    parser.add_argument("--model-out", default="models/pairwise_lgbm.txt", help="Path to output model file")
    parser.add_argument("--config-out", default="models/config.json", help="Path to output config file")
    parser.add_argument("--samples", type=int, default=50000, help="Number of S1 entities to sample for training")
    parser.add_argument("--max-cands", type=int, default=25, help="Max candidates per entity")
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        output_model_path=args.model_out,
        output_config_path=args.config_out,
        sample_entities=args.samples,
        max_cands=args.max_cands
    )


if __name__ == "__main__":
    main()
