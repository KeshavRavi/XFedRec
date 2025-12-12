# models/transformer_enc.py
"""
Transformer encoder-based recommender (lightweight).

This model treats item sequences as tokens and produces a user representation
via a Transformer encoder. For simplicity, the architecture is small.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderRec(nn.Module):
    def __init__(self, n_items, emb_size=64, n_heads=4, n_layers=2, max_seq_len=50, ff_dim=128, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_size, padding_idx=0)  # 0 as PAD
        self.pos_emb = nn.Embedding(max_seq_len, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_size, 1)

    def forward(self, item_seq):
        """
        item_seq: LongTensor (batch, seq_len)
        returns: predicted score for next-item (batch,)
        A simple approach: mean pool transformer outputs and map to scalar.
        """
        positions = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = x.permute(1,0,2)  # transformer expects seq_len, batch, emb
        h = self.transformer(x)  # seq_len, batch, emb
        h = h.permute(1,0,2)  # batch, seq_len, emb
        pooled = h.mean(dim=1)
        out = self.fc(pooled)
        return out.squeeze(-1)
