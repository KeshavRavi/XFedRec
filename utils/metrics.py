# utils/metrics.py
"""
Evaluation metrics for recommendation and regression tasks.

Includes:
- NDCG@k
- Hit Rate (HR) @k
- Precision@k
- RMSE, MAE
"""

import numpy as np

def precision_at_k(ranked_items, ground_truth, k):
    """
    ranked_items: list of item ids in predicted order
    ground_truth: set of relevant item ids
    """
    topk = ranked_items[:k]
    return len([i for i in topk if i in ground_truth]) / k

def hit_rate_at_k(ranked_items, ground_truth, k):
    topk = ranked_items[:k]
    return 1.0 if any(i in ground_truth for i in topk) else 0.0

def dcg_at_k(ranked_items, ground_truth, k):
    dcg = 0.0
    for idx, item in enumerate(ranked_items[:k]):
        rel = 1.0 if item in ground_truth else 0.0
        dcg += (2**rel - 1) / np.log2(idx + 2)
    return dcg

def idcg_at_k(ground_truth, k):
    # ideal DCG: all relevant items are in top positions
    rels = [1.0] * min(len(ground_truth), k)
    idcg = 0.0
    for idx, rel in enumerate(rels):
        idcg += (2**rel - 1) / np.log2(idx + 2)
    return idcg

def ndcg_at_k(ranked_items, ground_truth, k):
    idcg = idcg_at_k(ground_truth, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_items, ground_truth, k) / idcg

def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(np.mean((predictions - targets)**2))

def mae(predictions, targets):
    return np.mean(np.abs(np.array(predictions) - np.array(targets)))
