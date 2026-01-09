"""
Drift simulation utilities for federated clients.
"""

import pandas as pd
import numpy as np

def simulate_sudden_drift(df, drift_round, current_round):
    """
    Sudden drift: abruptly change item distribution.
    """
    if current_round < drift_round:
        return df

    # Shuffle item_ids aggressively
    df = df.copy()
    df["item_id"] = np.random.permutation(df["item_id"].values)
    return df


def simulate_incremental_drift(df, drift_start, current_round, strength=0.1):
    """
    Incremental drift: gradually perturb item preferences.
    """
    if current_round < drift_start:
        return df

    df = df.copy()
    n = int(len(df) * strength)
    idx = np.random.choice(len(df), n, replace=False)
    df.loc[idx, "item_id"] = np.random.randint(
        df["item_id"].min(),
        df["item_id"].max(),
        size=n
    )
    return df
