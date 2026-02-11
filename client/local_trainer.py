# client/local_trainer.py
"""
Local training loop for client.
This is a simplified trainer for demonstration.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange

class LocalTrainer:
    def __init__(self, model, device='cpu', config=None):
        self.model = model
        self.device = device
        self.config = config or {}
        self.epochs = self.config.get('local_epochs', 1)
        self.batch_size = self.config.get('local_batch_size', 64)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('lr', 1e-3))

    def _make_dataloader(self, local_df):
        """
        Construct a torch DataLoader from the local pandas DataFrame.
        Expects columns: user_id, item_id, rating
        """
        import torch.utils.data as data
        dataset = []
        for _, row in local_df.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id'])
            r = float(row.get('rating', 1.0))
            dataset.append((u, i, r))
        if len(dataset) == 0:
            # empty dataset fallback
            return None
        class SimpleDS(data.Dataset):
            def __init__(self, arr):
                self.arr = arr
            def __len__(self):
                return len(self.arr)
            def __getitem__(self, idx):
                u,i,r = self.arr[idx]
                return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(r, dtype=torch.float)
        return data.DataLoader(SimpleDS(dataset), batch_size=self.batch_size, shuffle=True)

    def train(self, local_df, epochs=None):
        epochs = epochs or self.epochs
        dl = self._make_dataloader(local_df)
        if dl is None:
            return 0.0 #return loss safely
        self.model.train()
        #track loss for drift detection
        total_loss=0.0
        count=0
        for e in range(epochs):
            for u,i,r in dl:
                u = u.to(self.device)
                i = i.to(self.device)
                r = r.to(self.device)
                pred = self.model(u, i)
                loss = self.criterion(pred, r)
                self.optimizer.zero_grad()
                loss.backward()
                # Optionally clip gradients using dp config (handled at client)
                self.optimizer.step()
                #accumulate loss
                total_loss += loss.item()
                count += 1
        #Return average training loss
        return total_loss / max(count, 1)
