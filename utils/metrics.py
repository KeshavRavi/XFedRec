# utils/metrics.py
import math
import numpy as np
import torch

def calculate_hr_ndcg(model, test_loader, n_items, k=10, device='cpu'):
    """
    Evaluates HR@K and NDCG@K using Leave-One-Out + 99 Negatives.
    """
    model.eval()
    hits = 0
    ndcgs = 0
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (user, item, rating)
            users, items, ratings = batch
            users = users.to(device)
            pos_items = items.to(device)
            
            # Process each user in the batch individually for ranking
            for idx in range(len(users)):
                u = users[idx].item()
                gt_item = pos_items[idx].item()
                
                # 1. Negative Sampling
                # Pick 99 random items the user hasn't interacted with (simplified: just random != gt)
                negatives = []
                while len(negatives) < 99:
                    n = np.random.randint(0, n_items)
                    if n != gt_item:
                        negatives.append(n)
                
                # 2. Prepare Batch: [Positive, Neg1, Neg2, ... Neg99]
                eval_items = [gt_item] + negatives
                eval_users = [u] * 100
                
                t_u = torch.tensor(eval_users, dtype=torch.long, device=device)
                t_i = torch.tensor(eval_items, dtype=torch.long, device=device)
                
                # 3. Predict Scores
                # Note: NCF outputs (batch,), ensure shape matches
                predictions = model(t_u, t_i).view(-1)
                
                # 4. Rank
                # We want the Top-K scores. 
                # The ground truth item is at index 0.
                _, indices = torch.topk(predictions, k)
                recommends = indices.cpu().tolist()
                
                # 5. Calculate Metrics
                if 0 in recommends:
                    hits += 1
                    # Rank is 0-indexed, so +2 for log formula (rank 1 -> index 0 -> log2(2))
                    rank = recommends.index(0)
                    ndcgs += 1 / math.log2(rank + 2)
                
                count += 1
                
                # OPTIMIZATION: Limit evaluation to speed up rounds
                # Evaluate max 100 users per client per round
                if count >= 100: break
            
            if count >= 100: break

    return {
        "HR@"+str(k): hits / count if count > 0 else 0.0,
        "NDCG@"+str(k): ndcgs / count if count > 0 else 0.0
    }