"""
utils/norms.py

Utilities for norm-based filtering and clipping of model updates.
"""

import torch

def l2_norm(state_dict):
    """Compute L2 norm of a model update."""
    return torch.sqrt(
        sum(torch.sum(v.float() ** 2) for v in state_dict.values())
    )

def clip_by_l2(state_dict, max_norm):
    """
    Clip update to maximum L2 norm.
    """
    norm = l2_norm(state_dict)
    if norm <= max_norm:
        return state_dict

    scale = max_norm / (norm + 1e-6)
    return {k: v * scale for k, v in state_dict.items()}
