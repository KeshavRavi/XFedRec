"""
Drift simulation utilities for federated clients.
"""

import pandas as pd
import numpy as np

def simulate_sudden_drift(df, drift_round, current_round, ratio=0.4, rating_noise=2.0):
    """
    Sudden drift: abruptly change item distribution.
    Happens at drift_round and persists afterwards.
    """
    if current_round < drift_round:
        return df

    # Work on a copy (VERY IMPORTANT)
    df = df.copy().reset_index(drop=True)

    n = int(len(df) * ratio)
    idx = np.random.choice(len(df), n, replace=False)

    # change items
    items = df["item_id"].unique()
    df.loc[idx, "item_id"] = np.random.choice(items, size=n, replace=True)

    # change ratings so loss shifts noticeably
    if "rating" in df.columns:
        noise = np.random.normal(0, rating_noise, size=n)
        df.loc[idx, "rating"] = df.loc[idx, "rating"] + noise
        df["rating"] = df["rating"].clip(1.0, 5.0)

    return df


def simulate_incremental_drift(df, drift_start, current_round, strength=0.1):
    """
    Incremental drift: gradually perturb item preferences.
    """
    if current_round < drift_start:
        return df

    df = df.copy().reset_index(drop=True)

    n = int(len(df) * strength)
    idx = np.random.choice(len(df), n, replace=False)

    items = df["item_id"].unique()
    df.loc[idx, "item_id"] = np.random.choice(items, size=n, replace=True)

    return df