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
    def __init__(self, client_id, local_data, model_cfg, device='cpu'):
        self.client_id = client_id
        self.local_data = local_data
        self.device = device
        self.model_cfg = model_cfg
        self.model = self._init_model(model_cfg).to(device)
        self.trainer = LocalTrainer(self.model, device=device, config=model_cfg)
        self.global_state = None
        self.current_round_df = self.local_data
        self.sudden_drift_done = False


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
        df = self.local_data
        drift_cfg = self.model_cfg.get("drift", {})

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
                    print(f"[Client {self.client_id}] Sudden drift triggered at round {round_id}")

            elif drift_cfg.get("type") == "incremental":
                df = simulate_incremental_drift(
                    df=df,
                    drift_start=drift_cfg.get("round", 2),
                    current_round=round_id
                )

        self.current_round_df = df    
        # 1️ Local training
        train_loss=self.trainer.train(
            df,
            epochs=self.model_cfg.get('local_epochs', 1)
        )
        self.last_train_loss = train_loss

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
        dp_cfg = self.model_cfg.get('dp', {})
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
        import numpy as np

        #  NEW: use drifted df for this round if available
        eval_df = getattr(self, "current_round_df", self.local_data)

        if eval_df is None or len(eval_df) == 0:
            return {'loss': 0.0}

        preds, targets = [], []
        for _, row in eval_df.sample(n=min(20, len(eval_df))).iterrows():
            u = torch.tensor([int(row['user_id'])], dtype=torch.long)
            i = torch.tensor([int(row['item_id'])], dtype=torch.long)
            with torch.no_grad():
                out = self.model(u, i)
            preds.append(out.item())
            targets.append(float(row.get('rating', 1.0)))

        mse = ((np.array(preds) - np.array(targets))**2).mean()
        return {'loss': float(mse)}
    def adapt_to_drift(self):
        """
        Adapt client model after drift detection (safer version).
        """
        print(f"[Client {self.client_id}] Adapting to drift")

        # Reduce LR of the REAL optimizer used in LocalTrainer
        for g in self.trainer.optimizer.param_groups:
            g["lr"] *= 0.5

        # Reset final layer weights (support both names)
        if hasattr(self.model, "output") and hasattr(self.model.output, "weight"):
            torch.nn.init.xavier_uniform_(self.model.output.weight)
            if getattr(self.model.output, "bias", None) is not None:
                torch.nn.init.zeros_(self.model.output.bias)

        if hasattr(self.model, "output_layer") and hasattr(self.model.output_layer, "weight"):
            torch.nn.init.xavier_uniform_(self.model.output_layer.weight)
            if getattr(self.model.output_layer, "bias", None) is not None:
                torch.nn.init.zeros_(self.model.output_layer.bias)

        # Boost epochs safely (avoid KeyError / non-int values)
        self.model_cfg["local_epochs"] = int(self.model_cfg.get("local_epochs", 1)) + 1

