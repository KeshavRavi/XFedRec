# data/loader.py

import pandas as pd
from typing import Dict

def load_movielens_100k(path_file: str) -> pd.DataFrame:
    """
    Loads the converted MovieLens 100K CSV.
    Expects columns: user_id, item_id, rating
    """
    df = pd.read_csv(path_file)
    return df


def dataset_to_user_item_lists(df) -> Dict[int, list]:
    """
    Convert DataFrame to dict: user -> list of (item, rating)
    No timestamp used because converted CSV does not contain it.
    """
    users = {}
    for _, row in df.iterrows():
        u = int(row['user_id'])
        i = int(row['item_id'])
        r = float(row['rating'])
        users.setdefault(u, []).append((i, r))
    return users
