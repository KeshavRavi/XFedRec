import random

def random_selection(clients, frac):
    """
    Randomly select a fraction of clients.
    Ensures at least one client is selected.
    """
    if clients is None or len(clients) == 0:
        return []

    k = max(1, int(frac * len(clients)))
    return random.sample(clients, k)
