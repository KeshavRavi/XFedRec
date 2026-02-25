# data/partition.py
"""
Non-IID partitioning utilities for simulating clients.

Supports:
- user-based partitioning (each user -> client)
- time-based split (simulate drift by assigning older/newer data to different clients)
- category-skew (requires item category information in dataset)
"""

import random
from collections import defaultdict
from typing import Dict, List

import pandas as pd

def partition_by_user(df, n_clients):
    """
    Partition by user: aggregate users into n_clients groups.
    Returns list of DataFrames (one per client).
    """
    users = df['user_id'].unique().tolist()
    random.shuffle(users)
    groups = [set() for _ in range(n_clients)]
    for idx, u in enumerate(users):
        groups[idx % n_clients].add(u)
    client_dfs = []
    for g in groups:
        client_dfs.append(df[df['user_id'].isin(g)].copy())
    return client_dfs

def partition_by_time(df, n_clients, timestamp_col='timestamp'):
    """
    Sort by time and partition into n_clients chronological chunks.
    Useful to simulate temporal drift.
    """
    df_sorted = df.sort_values(timestamp_col)
    total = len(df_sorted)
    chunk = total // n_clients
    client_dfs = []
    for i in range(n_clients):
        start = i * chunk
        end = (i+1) * chunk if i < n_clients - 1 else total
        client_dfs.append(df_sorted.iloc[start:end].copy())
    return client_dfs

def partition_by_category_skew(df, n_clients, category_col='category', skew=0.7):
    """
    Create skewed clients where each client has majority from a particular category.
    skew: fraction of client data from dominant category.
    """
    categories = df[category_col].unique().tolist()
    client_dfs = []
    for i in range(n_clients):
        dom = categories[i % len(categories)]
        dom_df = df[df[category_col] == dom]
        other_df = df[df[category_col] != dom]
        dom_count = int(skew * len(dom_df))
        selected_dom = dom_df.sample(n=min(dom_count, len(dom_df)))
        selected_other = other_df.sample(n=max(1, int((1 - skew) * len(selected_dom))))
        client_dfs.append(pd.concat([selected_dom, selected_other]).sample(frac=1.0).reset_index(drop=True))
    return client_dfs

def partition_non_iid_dirichlet(df, n_clients, alpha=0.1):
    """
    Partition data non-IID across clients using a Dirichlet distribution over items.
    Lower alpha = higher skew (more non-IID).
    """
    import numpy as np
    import pandas as pd
    
    client_dfs = [[] for _ in range(n_clients)]
    
    # Group by item to create skew in what items clients interact with
    groups = df.groupby('item_id')
    
    for item_id, group_df in groups:
        # Sample proportions for this item across all clients
        # e.g., [0.9, 0.05, 0.05...] means Client 0 gets 90% of interactions for this item
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        
        # Convert proportions to actual row counts
        n_interactions = len(group_df)
        counts = np.random.multinomial(n_interactions, proportions)
        
        # Split the group_df according to counts
        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                client_dfs[client_id].append(group_df.iloc[start:start+count])
                start += count
                
    # Concat lists of dfs back into single DataFrames and shuffle
    final_dfs = []
    for c_dfs in client_dfs:
        if len(c_dfs) > 0:
            final_dfs.append(pd.concat(c_dfs).sample(frac=1.0).reset_index(drop=True))
        else:
            # Failsafe for an empty client
            final_dfs.append(pd.DataFrame(columns=df.columns))
            
    return final_dfs