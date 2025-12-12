# utils/dp.py
"""
Differential Privacy utilities.

Provides:
- gradient/weight clipping
- Gaussian noise addition with configurable sigma

This module is intentionally simple and suitable for experiments.
For production-level DP use Opacus or TensorFlow Privacy.
"""

import numpy as np
import torch

def clip_gradients(model, max_norm):
    """Clips gradients of model parameters in-place."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def add_gaussian_noise_to_tensor(tensor, sigma, seed=None):
    """Add Gaussian noise to a torch tensor and return new tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.normal(mean=0.0, std=sigma, size=tensor.shape, device=tensor.device)
    return tensor + noise

def add_dp_to_state_dict(state_dict, sigma, seed=None):
    """
    Add Gaussian noise to all tensors in a state_dict (weights).
    Use for local model update perturbation for DP-SGD-like effect.
    """
    noisy = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            noisy[k] = add_gaussian_noise_to_tensor(v, sigma, seed)
        else:
            noisy[k] = v
    return noisy
