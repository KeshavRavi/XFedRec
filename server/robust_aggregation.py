import torch
import numpy as np
from copy import deepcopy

def flatten_state(state_dict):
    return torch.cat([v.flatten() for v in state_dict.values()])

def unflatten_state(flat_tensor, ref_state):
    new_state = {}
    idx = 0
    for k, v in ref_state.items():
        numel = v.numel()
        new_state[k] = flat_tensor[idx:idx+numel].view(v.shape)
        idx += numel
    return new_state


# ---------- MEDIAN ----------
def median_aggregate(states):
    flat = torch.stack([flatten_state(s) for s in states])
    median = torch.median(flat, dim=0).values
    return unflatten_state(median, states[0])


# ---------- KRUM ----------
def krum_aggregate(states, f=1):
    flats = [flatten_state(s) for s in states]
    n = len(flats)
    scores = []

    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(torch.norm(flats[i] - flats[j])**2)
        dists.sort()
        scores.append(sum(dists[:n - f - 2]))

    krum_idx = int(torch.argmin(torch.tensor(scores)))
    return deepcopy(states[krum_idx])


# ---------- FEDQ ----------
def fedq_aggregate(states, qualities):
    weights = torch.tensor(qualities)
    weights = weights / weights.sum()

    flat = torch.stack([flatten_state(s) for s in states])
    agg = torch.sum(flat * weights[:, None], dim=0)
    return unflatten_state(agg, states[0])
