# server/selection.py
"""
Client selection strategies.
"""

import random

def random_selection(clients, frac=0.5, seed=None):
    """
    Randomly select fraction of clients.
    """
    if seed is not None:
        random.seed(seed)
    k = max(1, int(len(clients) * frac))
    return random.sample(clients, k)
