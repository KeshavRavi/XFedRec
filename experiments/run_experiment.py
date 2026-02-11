# experiments/run_experiment.py
"""
Main entrypoint for running experiments.

This script configures a federated run and demonstrates:
- Building server and clients
- Running a few federated rounds
Note: This is a skeleton demonstrating the flow; extend it to run full experiments.
"""

import argparse
import yaml
import os
import random
import torch
from server.server import Server
from client.client import Client
from models.ncf import NeuralCF
from models.transformer_enc import TransformerEncoderRec
from models.fed_dae import FedDAE
from data.loader import dataset_to_user_item_lists, load_movielens_100k
from data.partition import partition_by_user, partition_by_time
from xai.lime_shap_wrapper import XAIExplainer
def run_experiment(config_path='experiments/scripts/experiment_config.yaml', return_server=False):
    """
    Build clients and server, run federated rounds.
    If return_server=True, return the server object for XAI or further inspection.
    """
    # Load config
    cfg = load_config(config_path)
    
    # Load dataset
    df = load_movielens_100k(cfg['data']['movielens_path'])

    # Build model cfg
    model_cfg = cfg['model']
    model_cfg["drift"] = cfg.get("drift", {"enabled": False})


    # FIX: make embedding sizes safe (prevents index out of range)
    model_cfg["n_users"] = int(df["user_id"].max()) + 1
    model_cfg["n_items"] = int(df["item_id"].max()) + 1
    print("[Config] n_users =", model_cfg["n_users"], "n_items =", model_cfg["n_items"])
    # Partition dataset among clients
    if cfg.get("drift", {}).get("enabled", False) and "timestamp" in df.columns:
        partitions = partition_by_time(df, cfg['federation']['n_clients'], timestamp_col="timestamp")
    else:
        partitions = partition_by_user(df, cfg['federation']['n_clients'])



    
    # Build clients
    clients = build_clients_from_partitions(partitions, model_cfg, device=cfg.get('device', 'cpu'))
    
    # Build server
    server = Server(clients=clients, config=cfg)
    
    # Run federated rounds
    metrics = server.run_rounds(
        num_rounds=cfg['federation']['rounds']
    )

    # ===== Step 5.3: Save metrics =====
    os.makedirs("experiments/results", exist_ok=True)

    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    csv_path = f"experiments/results/{exp_name}.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print(f"[Experiment] Results saved to {csv_path}")

    if return_server:
        return server

    return metrics

def load_config(path):
    with open(path,encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_clients_from_partitions(partitions, model_cfg, device='cpu'):
    clients = []
    for cid, df in enumerate(partitions):
        client = Client(client_id=cid, local_data=df, model_cfg=model_cfg, device=device)
        clients.append(client)
    return clients

def main(config_path):
    # Just call run_experiment; old command-line behavior unchanged
    run_experiment(config_path=config_path, return_server=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/scripts/experiment_config.yaml')
    args = parser.parse_args()
    main(args.config)
