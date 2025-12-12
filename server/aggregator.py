# server/aggregator.py
"""
Robust aggregation strategies:
- simple average (FedAvg)
- coordinate-wise median
- Krum (simple implementation)
- FedQ placeholder (quantile-based robust aggregator)
"""

import numpy as np
import torch
import copy

def fedavg(updates):
    """
    updates: list of state_dicts (tensors)
    returns: aggregated state_dict
    """
    agg = {}
    n = len(updates)
    for k in updates[0].keys():
        agg[k] = sum([u[k].float() for u in updates]) / n
    return agg

def coordinate_median(updates):
    """
    Compute coordinate-wise median across flattened tensors per key.
    """
    agg = {}
    for k in updates[0].keys():
        stacked = torch.stack([u[k].float().view(-1) for u in updates], dim=1)  # (dim, n)
        med = torch.median(stacked, dim=1)[0]
        agg[k] = med.view_as(updates[0][k])
    return agg

def krum(updates, f=1):
    """
    Simple Krum implementation:
    - Flatten each model to a vector
    - Compute pairwise distances, score each update by sum of distances to closest n-f-2 updates
    - Pick update with minimal score (single model)
    NOTE: Krum returns a single model. For fairness, we return that model as aggregated.
    """
    n = len(updates)
    vecs = []
    for u in updates:
        vec = torch.cat([v.view(-1).cpu() for v in u.values()])
        vecs.append(vec.numpy())
    vecs = np.stack(vecs)
    dists = np.sum((vecs[:,None,:] - vecs[None,:,:])**2, axis=2)
    scores = []
    for i in range(n):
        d = np.sort(dists[i])
        nb = n - f - 2
        nb = max(0, nb)
        scores.append(d[1:1+nb].sum() if nb > 0 else 0.0)
    best = int(np.argmin(scores))
    return updates[best]

def fedq_placeholder(updates):
    """
    Placeholder for FedQ robust aggregator. For now, falls back to median.
    """
    return coordinate_median(updates)
