# client/client.py

import torch
import copy
from .local_trainer import LocalTrainer
from utils.dp import add_dp_to_state_dict
from data.drift_simulator import (
    simulate_sudden_drift,
    simulate_incremental_drift
)

class Client:
    def __init__(self, client_id, local_data, model_cfg, drift_cfg=None, dp_cfg=None, device='cpu'):
        self.client_id = client_id
        self.local_data = local_data
        self.device = device

        self.model_cfg = model_cfg
        self.drift_cfg = drift_cfg or {}   #  drift config comes from YAML top-level
        self.dp_cfg = dp_cfg or {}         #  dp config from YAML top-level

        self.model = self._init_model(model_cfg).to(device)
        self.trainer = LocalTrainer(self.model, device=device, config=model_cfg)

        self.global_state = None
        self.current_round_df = self.local_data
        self.sudden_drift_done = False
        self.drifted_data = None


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
        return copy.deepcopy(self.model.state_dict())

    def set_global_model(self, state_dict):
        self.global_state = state_dict
        self.model.load_state_dict(state_dict)

    def local_update(self, round_id=0):
        """
        Perform local training and return local model update.
        """
        # 0️ Drift Simulation
        df = self.drifted_data if self.drifted_data is not None else self.local_data
        drift_cfg = self.drift_cfg

        if drift_cfg.get("enabled", False):
            
            if drift_cfg.get("type") == "sudden":
                drift_round=drift_cfg.get("round", 3)
                
                #  apply sudden drift ONCE per client (first time it reaches drift_round)
                if (not self.sudden_drift_done) and (round_id >= drift_round):    
                    df = simulate_sudden_drift(
                        df=df,
                        drift_round=drift_round,
                        current_round=round_id
                    )
                    self.sudden_drift_done = True
                    self.drifted_data = df   # SAVE drifted df
                    print(f"\n[Client {self.client_id}]   SUDDEN DRIFT INJECTED at Round {round_id} \n")

            elif drift_cfg.get("type") == "incremental":
                drift_start=drift_cfg.get("round", 2)
                if round_id >= drift_start:
                    df = simulate_incremental_drift(
                        df=df,
                        drift_start=drift_start,
                        current_round=round_id
                    )
                    self.drifted_data = df   # SAVE updated df every round (incremental persists)

        self.current_round_df = df    
        df_train = df.copy()
        if int(df_train["user_id"].min()) == 1:
            df_train["user_id"] = df_train["user_id"] - 1
        if int(df_train["item_id"].min()) == 1:
            df_train["item_id"] = df_train["item_id"] - 1

        # 1️ Local training
        train_loss=self.trainer.train(df_train,epochs=self.model_cfg.get('local_epochs', 1))
        self.current_round_df = df_train
        self.drifted_data = df_train if self.drifted_data is not None else None

        # 2️  Client-side XAI 
        if self.model_cfg.get("enable_xai", False):
            from .xai_local import LocalXAI

            xai = LocalXAI(
                model=self.model,
                feature_names=["user_id", "item_id"],
                device=self.device
            )

            # background data for explanation
            bg = self.local_data[["user_id", "item_id"]].values[:50]
            if len(bg) > 0:
                instance = bg[0]

                xai.explain_instance(
                    background_data=bg,
                    instance=instance,
                    client_id=self.client_id,
                    round_id=round_id
                )

        # 3️ Prepare model update
        state = copy.deepcopy(self.model.state_dict())

        # 4️ Optional DP
        dp_cfg = self.dp_cfg
        if dp_cfg.get('enabled', False):
            state = add_dp_to_state_dict(
                state,
                sigma=dp_cfg.get('sigma', 0.1)
            )

        return state,{
            "train_loss": train_loss,
            "n": len(df)
        }

    def update_server_version(self, server_state):
        self.model.load_state_dict(server_state)

    def evaluate_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()
        import numpy as np
        
        # 1. Determine Data Source (Drifted vs Normal)
        eval_df = getattr(self, "current_round_df", None)
        if eval_df is None:
            eval_df = self.local_data

        if eval_df is None or len(eval_df) == 0:
            return {'loss': 0.0, 'hr': 0.0, 'ndcg': 0.0}

        # 2. MSE Calculation (Keep this for Drift Detection stability)
        preds, targets = [], []
        # Sample smaller set for MSE to keep it fast
        sample_df = eval_df.sample(n=min(200, len(eval_df)))
        
        with torch.no_grad():
            for _, row in sample_df.iterrows():
                u_id = self._idx(sample_df, "user_id", row["user_id"])
                i_id = self._idx(sample_df, "item_id", row["item_id"])
                
                u = torch.tensor([u_id], dtype=torch.long, device=self.device)
                i = torch.tensor([i_id], dtype=torch.long, device=self.device)
                
                out = self.model(u, i)
                preds.append(out.item())
                targets.append(float(row.get('rating', 1.0)))

        mse = ((np.array(preds) - np.array(targets)) ** 2).mean() if preds else 0.0

        # 3. Ranking Metrics (HR / NDCG)
        from utils.metrics import calculate_hr_ndcg
        
        # Use the LAST 20% of data as a "Held-out Test Set" for this round
        n_val = int(len(eval_df) * 0.2)
        if n_val > 0:
            val_df = eval_df.iloc[-n_val:]
            val_loader = self.trainer._make_dataloader(val_df)
            
            # Pass n_items from config to avoid model dependency issues
            n_items = self.model_cfg.get("n_items", 1683)
            
            ranking_metrics = calculate_hr_ndcg(
                self.model, 
                val_loader, 
                n_items=n_items, 
                k=10, 
                device=self.device
            )
        else:
            ranking_metrics = {"HR@10": 0.0, "NDCG@10": 0.0}

        return {
            'loss': float(mse),
            'hr': ranking_metrics["HR@10"],
            'ndcg': ranking_metrics["NDCG@10"]
        }
    
    
    def adapt_to_drift(self):
        """
        Adapt client model after drift detection.
        Strategy: Reset Head + Increase Plasticity + Clear Optimizer Momentum.
        """
        print(f"[Client {self.client_id}]  Adapting to drift: Resetting head & Optimizer")

        # 1. Reset Final Layer
        if hasattr(self.model, "output") and hasattr(self.model.output, "weight"):
            torch.nn.init.xavier_uniform_(self.model.output.weight)
            if self.model.output.bias is not None:
                torch.nn.init.zeros_(self.model.output.bias)

        # 2. Aggressive Plasticity Boost
        current_epochs = int(self.model_cfg.get("local_epochs", 1))
        self.model_cfg["local_epochs"] = current_epochs + 2 
        
        # 3. Reset Optimizer State
        # FIX: Use self.model_cfg instead of self.config
        lr = float(self.model_cfg.get('lr', 1e-3)) * 1.5
        
        self.trainer.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr
        )
    
    def _idx(self, df, col, value):
        # if data starts at 1 (MovieLens), convert to 0-based
        # if it already starts at 0, keep as-is
        if df is not None and len(df) > 0 and int(df[col].min()) == 1:
            return int(value) - 1
        return int(value)


