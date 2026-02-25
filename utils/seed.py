# utils/seed.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Locks all random number generators to ensure 100% reproducibility.
    """
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. Python hash seed (affects dictionary ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
    # 5. PyTorch CUDNN backend (forces deterministic algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Setup] 🔒 Global random seed locked to: {seed}")