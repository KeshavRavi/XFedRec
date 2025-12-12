# models/fed_dae.py
"""
FedDAE-style autoencoder for collaborative filtering (simplified).

This autoencoder takes a sparse user-item vector and reconstructs ratings.
Useful for federated setting where clients have user-rating vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FedDAE(nn.Module):
    def __init__(self, n_items, hidden_sizes=[512,256], dropout=0.2):
        super().__init__()
        layers = []
        input_dim = n_items
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        self.encoder = nn.Sequential(*layers)
        # decoder mirror
        hidden_sizes.reverse()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, n_items))  # reconstruct ratings
        self.decoder = nn.Sequential(*layers)

    def forward(self, user_vec):
        """
        user_vec: dense vector (batch, n_items) representing ratings (0 for unknown)
        """
        z = self.encoder(user_vec)
        recon = self.decoder(z)
        return recon
