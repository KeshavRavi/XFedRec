# utils/compression.py
"""
Compression placeholder module.

Provides:
- identity compressor
- simple quantization compressor
- placeholder for DeepCABAC (not implemented)
"""

import numpy as np
import torch

def identity_compress(tensor):
    """No compression, just returns bytes via numpy."""
    return tensor.cpu().numpy()

def simple_quantize(tensor, bits=8):
    """
    Uniform quantization to simulate compression.
    Returns quantized numpy array and scale for dequantization.
    """
    x = tensor.cpu().numpy()
    minv = x.min()
    maxv = x.max()
    if maxv == minv:
        return x, (minv, maxv)
    q_levels = 2**bits - 1
    q = np.round((x - minv) / (maxv - minv) * q_levels).astype(np.uint8)
    return q, (minv, maxv)

def simple_dequantize(q, scale):
    minv, maxv = scale
    q_levels = 255
    return (q.astype(np.float32) / q_levels) * (maxv - minv) + minv

def sparsify_update(update_dict, compression_rate=0.8):
    """
    Applies Top-K sparsification to the model update.
    compression_rate=0.8 means drop 80% of the weights, keep only the top 20% largest magnitudes.
    """
    if compression_rate <= 0.0 or compression_rate >= 1.0:
        return update_dict # No compression

    compressed_update = {}

    for name, param in update_dict.items():
        # Only compress floating point weights (ignore integer metadata)
        if not param.is_floating_point():
            compressed_update[name] = param
            continue

        param_flat = param.view(-1)
        n_elements = param_flat.numel()

        # Calculate how many parameters to KEEP
        k = max(1, int(n_elements * (1.0 - compression_rate)))

        if k >= n_elements:
            compressed_update[name] = param
            continue

        # Find the threshold value for the Top-K largest absolute magnitudes
        threshold, _ = torch.topk(torch.abs(param_flat), k)
        kth_largest = threshold[-1]

        # Create a boolean mask: True if abs(weight) >= threshold
        mask = torch.abs(param) >= kth_largest

        # Zero out the weights that didn't make the cut
        compressed_param = param * mask.to(param.dtype)
        compressed_update[name] = compressed_param

    return compressed_update