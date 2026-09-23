from __future__ import annotations

import numpy as np


def aggregate_video(scores, method="mean", top_k=3, threshold=0.5):
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must contain at least one value")
    method = method.lower()
    if method == "mean":
        return float(values.mean())
    if method in {"top-k", "top_k"}:
        count = min(max(int(top_k), 1), values.size)
        return float(np.sort(values)[-count:].mean())
    if method == "median":
        return float(np.median(values))
    if method in {"majority", "majority-vote", "majority_vote"}:
        return float((values >= threshold).mean() >= 0.5)
    raise ValueError("method must be mean, top-k, median, or majority")


def compare_aggregations(scores_by_video, labels, threshold=0.5, top_k=3):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    results = []
    for method in ("mean", "top-k", "median", "majority"):
        scores = [
            aggregate_video(scores, method, top_k=top_k, threshold=threshold)
            for scores in scores_by_video
        ]
        predictions = (np.asarray(scores) >= threshold).astype(int)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")
        results.append({
            "method": method,
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, zero_division=0),
            "auc": auc,
        })
    return results
