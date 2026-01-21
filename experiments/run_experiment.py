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
from data.partition import partition_by_user
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
    
    # Partition dataset among clients
    partitions = partition_by_user(df, cfg['federation']['n_clients'])
    
    # Build model cfg
    model_cfg = cfg['model']
    
    # Build clients
    clients = build_clients_from_partitions(partitions, model_cfg, device=cfg.get('device', 'cpu'))
    
    # Build server
    server = Server(clients=clients, config=cfg)
    
    # Run federated rounds
    metrics =server.run_rounds(
        num_rounds=cfg['federation']['rounds']
    )
    return metrics
    
    
    if return_server:
        return server

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
