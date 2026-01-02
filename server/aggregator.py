"""
server/aggregator.py

Implements robust aggregation strategies for federated learning.

Supported methods:
- FedAvg: Standard federated averaging
- Coordinate-wise Median: Robust to extreme updates
- Krum: Byzantine-resilient aggregation
- FedQ (placeholder): Quantile-based robust aggregation (fallback to median)

All functions operate on PyTorch state_dicts.
"""

import numpy as np
import torch
import copy


# --------------------------------------------------
# FedAvg
# --------------------------------------------------

def fedavg(updates):
    """
    Standard Federated Averaging.

    Args:
        updates (list): List of client model state_dicts

    Returns:
        dict: Aggregated global state_dict
    """
    assert len(updates) > 0, "No updates provided to FedAvg"

    agg = {}
    n = len(updates)

    for k in updates[0].keys():
        agg[k] = sum(u[k].float() for u in updates) / n

    return agg


# --------------------------------------------------
# Coordinate-wise Median
# --------------------------------------------------

def coordinate_median(updates):
    """
    Coordinate-wise median aggregation.

    Robust to outliers and poisoning attacks.

    Args:
        updates (list): List of client model state_dicts

    Returns:
        dict: Aggregated global state_dict
    """
    assert len(updates) > 0, "No updates provided to Median aggregation"

    agg = {}

    for k in updates[0].keys():
        # Stack flattened tensors across clients
        stacked = torch.stack(
            [u[k].float().view(-1) for u in updates],
            dim=1
        )  # shape: (num_params, num_clients)

        median_vals = torch.median(stacked, dim=1).values
        agg[k] = median_vals.view_as(updates[0][k])

    return agg


# --------------------------------------------------
# Krum Aggregation
# --------------------------------------------------

def krum(updates, f=1):
    """
    Krum aggregation (single-model selection).

    Selects the client update closest to the majority
    under the assumption of at most f malicious clients.

    Args:
        updates (list): List of client model state_dicts
        f (int): Maximum number of Byzantine (malicious) clients

    Returns:
        dict: Selected client model state_dict
    """
    assert len(updates) > 0, "No updates provided to Krum"

    n = len(updates)
    if n <= 2 * f + 2:
        # Not enough clients for Krum guarantees
        return fedavg(updates)

    # Flatten all model updates
    flat_updates = []
    for u in updates:
        flat = torch.cat([v.view(-1).cpu() for v in u.values()])
        flat_updates.append(flat.numpy())

    flat_updates = np.stack(flat_updates)  # shape: (n_clients, dim)

    # Compute pairwise squared Euclidean distances
    dist_matrix = np.sum(
        (flat_updates[:, None, :] - flat_updates[None, :, :]) ** 2,
        axis=2
    )

    # Compute Krum scores
    scores = []
    for i in range(n):
        sorted_dists = np.sort(dist_matrix[i])
        score = np.sum(sorted_dists[1 : n - f - 1])
        scores.append(score)

    best_idx = int(np.argmin(scores))
    return copy.deepcopy(updates[best_idx])


# --------------------------------------------------
# FedQ (Placeholder)
# --------------------------------------------------

def fedq_placeholder(updates):
    """
    Placeholder for FedQ aggregation.

    FedQ is a quality-aware quantile-based aggregator.
    For now, this falls back to coordinate-wise median.

    Args:
        updates (list): List of client model state_dicts

    Returns:
        dict: Aggregated global state_dict
    """
    return coordinate_median(updates)


# --------------------------------------------------
# Unified Aggregation Interface
# --------------------------------------------------

def aggregate(updates, method="fedavg", f=1):
    """
    Unified aggregation interface.

    Args:
        updates (list): List of client model state_dicts
        method (str): Aggregation method
                      ['fedavg', 'median', 'krum', 'fedq']
        f (int): Number of assumed Byzantine clients (for Krum)

    Returns:
        dict: Aggregated global state_dict
    """
    if method == "median":
        return coordinate_median(updates)

    elif method == "krum":
        return krum(updates, f=f)

    elif method == "fedq":
        return fedq_placeholder(updates)

    else:
        return fedavg(updates)
