# models/ncf.py
"""
Neural Collaborative Filtering (NCF) minimal implementation.

This is a compact example combining embeddings + MLP for implicit feedback ranking/regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, emb_size=32, mlp_layers=[64,32,16], dropout=0.2):
        """
        n_users, n_items: number of users and items
        emb_size: embedding size for GMF / MLP input
        mlp_layers: list of ints describing MLP hidden sizes
        """
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_items, emb_size)

        mlp_input = emb_size * 2
        mlp = []
        for hidden in mlp_layers:
            mlp.append(nn.Linear(mlp_input, hidden))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
            mlp_input = hidden
        self.mlp = nn.Sequential(*mlp)
        self.output = nn.Linear(mlp_input, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_indices, item_indices):
        """
        user_indices: tensor (batch,)
        item_indices: tensor (batch,)
        returns: predicted score (batch, 1)
        """
        u = self.user_emb(user_indices)
        i = self.item_emb(item_indices)
        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)
        out = self.output(x)
        return out.squeeze(-1)
