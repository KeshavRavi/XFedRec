# client/client.py
"""
Client class that holds local dataset, local model, and training routine.
This class communicates with Server via method calls in this simplified simulator.
"""

import torch
import torch.nn as nn
import copy
from .local_trainer import LocalTrainer
from utils.dp import add_dp_to_state_dict

class Client:
    def __init__(self, client_id, local_data, model_cfg, device='cpu'):
        """
        local_data: pandas DataFrame for this client
        model_cfg: dict to identify which model to initialize
        """
        self.client_id = client_id
        self.local_data = local_data
        self.device = device
        self.model_cfg = model_cfg
        self.model = self._init_model(model_cfg).to(device)
        self.trainer = LocalTrainer(self.model, device=device, config=model_cfg)
        self.global_state = None

    def _init_model(self, cfg):
        model_type = cfg.get('type', 'ncf')
        if model_type == 'ncf':
            from models.ncf import NeuralCF
            return NeuralCF(cfg['n_users'], cfg['n_items'], emb_size=cfg.get('emb_size', 32))
        elif model_type == 'transformer':
            from models.transformer_enc import TransformerEncoderRec
            return TransformerEncoderRec(cfg['n_items'], emb_size=cfg.get('emb_size', 64))
        elif model_type == 'fed_dae':
            from models.fed_dae import FedDAE
            return FedDAE(cfg['n_items'])
        else:
            raise ValueError("Unknown model type")

    def get_model_state(self):
        """Return state_dict copy."""
        return copy.deepcopy(self.model.state_dict())

    def set_global_model(self, state_dict):
        """Receive global model before training."""
        self.global_state = state_dict
        self.model.load_state_dict(state_dict)

    def local_update(self):
        """
        Perform local training using LocalTrainer and return local model state_dict.
        Optionally apply DP noise.
        """
        self.trainer.train(self.local_data, epochs=self.model_cfg.get('local_epochs', 1))
        state = copy.deepcopy(self.model.state_dict())
        # optionally add DP noise
        dp_cfg = self.model_cfg.get('dp', {})
        if dp_cfg.get('enabled', False):
            state = add_dp_to_state_dict(state, sigma=dp_cfg.get('sigma', 0.1))
        return state

    def update_server_version(self, server_state):
        """Apply server version (post-aggregation) if needed."""
        self.model.load_state_dict(server_state)

    def evaluate_model(self, state_dict):
        """Evaluate a model (given by state_dict) on local validation split."""
        self.model.load_state_dict(state_dict)
        # lightweight: compute dummy loss as mean squared error on available ratings
        import numpy as np
        if self.local_data is None or len(self.local_data) == 0:
            return {'loss': 0.0}
        preds = []
        targets = []
        for _, row in self.local_data.sample(n=min(20, len(self.local_data))).iterrows():
            u = torch.tensor([int(row['user_id'])], dtype=torch.long)
            i = torch.tensor([int(row['item_id'])], dtype=torch.long)
            with torch.no_grad():
                out = self.model(u, i) if hasattr(self.model, '__call__') else torch.tensor([0.0])
            preds.append(out.item() if hasattr(out, 'item') else float(out))
            targets.append(float(row.get('rating', 1.0)))
        mse = ((np.array(preds) - np.array(targets))**2).mean() if len(preds) else 0.0
        return {'loss': float(mse)}
