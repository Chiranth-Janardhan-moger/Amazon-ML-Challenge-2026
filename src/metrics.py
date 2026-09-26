"""
Official evaluation metric for Amazon ML Challenge 2026: Macro-averaged F0.5 Score.
Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Evaluated per Source 1 entity, macro-averaged across all Source 1 entities in the evaluation set.
Singletons (S1 with no matches):
    - Correctly predicting empty set: 1.0
    - Predicting any match: 0.0
"""

from typing import Dict, Set


def compute_entity_f_beta(pred_set: Set[str], gt_set: Set[str], beta: float = 0.5) -> float:
    """Calculate F_beta score for a single Source 1 entity."""
    n_gt = len(gt_set)
    n_pred = len(pred_set)

    # Singleton case (no true matches)
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0

    # Model predicted nothing for an entity that has true matches
    if n_pred == 0:
        return 0.0

    true_positives = len(pred_set & gt_set)
    if true_positives == 0:
        return 0.0

    precision = true_positives / n_pred
    recall = true_positives / n_gt

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0

    return ((1.0 + beta_sq) * precision * recall) / denom


def compute_entity_f05(pred_set: Set[str], gt_set: Set[str]) -> float:
    return compute_entity_f_beta(pred_set, gt_set, beta=0.5)


def macro_f_beta_score(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]], beta: float = 0.5) -> float:
    """Compute macro-averaged F_beta across all entities."""
    total_entities = len(ground_truth)
    if total_entities == 0:
        return 0.0

    total_score = 0.0
    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_score += compute_entity_f_beta(pred_set, gt_set, beta=beta)

    return total_score / total_entities


def evaluate_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> float:
    return macro_f_beta_score(ground_truth, predictions, beta=0.5)

