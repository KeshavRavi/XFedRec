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
