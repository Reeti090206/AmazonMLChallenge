"""
evaluate.py — F0.5 (macro-averaged per S1 entity) evaluation.

The official metric is:

    F_0.5 = (1.25 × P × R) / (0.25 × P + R)

macro-averaged across all Source 1 entities.  Singleton entities (zero
true matches) that are correctly predicted as empty receive F0.5 = 1.0.
"""

import numpy as np


def f05_single(predicted: set, truth: set) -> float:
    """Compute F0.5 for a single S1 entity.

    Args:
        predicted: Set of predicted matched entity IDs.
        truth: Set of true matched entity IDs (empty for singletons).

    Returns:
        F0.5 score in [0.0, 1.0].
    """
    # Singleton correctly predicted empty → perfect score
    if not truth and not predicted:
        return 1.0
    # Singleton but we predicted something → penalise
    if not truth and predicted:
        return 0.0
    # Non-singleton but we predicted nothing
    if truth and not predicted:
        return 0.0

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def evaluate_predictions(predictions: dict, ground_truth: dict) -> dict:
    """Macro-average F0.5 across all S1 entities.

    Args:
        predictions: {s1_id: set(predicted_ids)}  — every S1 must appear.
        ground_truth: {s1_id: set(true_ids)}  — empty set for singletons.

    Returns:
        dict with keys: macro_f05, macro_precision, macro_recall,
        n_entities, n_singletons, n_singleton_correct, n_singleton_wrong.
    """
    scores = []
    precisions = []
    recalls = []
    n_singletons = 0
    n_singleton_correct = 0
    n_singleton_wrong = 0

    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        truth_set = set(truth) if not isinstance(truth, set) else truth
        pred_set = set(pred) if not isinstance(pred, set) else pred

        if not truth_set:
            n_singletons += 1
            if not pred_set:
                n_singleton_correct += 1
            else:
                n_singleton_wrong += 1

        f = f05_single(pred_set, truth_set)
        scores.append(f)

        # Per-entity precision and recall
        if truth_set and pred_set:
            tp = len(pred_set & truth_set)
            p = tp / len(pred_set)
            r = tp / len(truth_set)
        elif not truth_set and not pred_set:
            p, r = 1.0, 1.0
        elif not truth_set and pred_set:
            p, r = 0.0, 1.0
        else:
            p, r = 1.0, 0.0
        precisions.append(p)
        recalls.append(r)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0

    return {
        "macro_f05": round(macro_f05, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "n_entities": len(scores),
        "n_singletons": n_singletons,
        "n_singleton_correct": n_singleton_correct,
        "n_singleton_wrong": n_singleton_wrong,
    }
