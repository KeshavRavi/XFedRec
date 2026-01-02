"""
Server orchestration for federated learning.
"""

import copy
import numpy as np
from .aggregator import aggregate
from .selection import random_selection
from .logger import ExperimentLogger
from utils.norms import clip_by_l2


class Server:
    def __init__(self, clients, config):
        """
        clients: list of Client objects
        config: experiment config dict
        """
        self.clients = clients
        self.config = config
        self.logger = ExperimentLogger()

        # Initialize global model from first client
        self.global_model = None
        if len(clients) > 0:
            self.global_model = copy.deepcopy(clients[0].get_model_state())
        print(f"[Server] Total clients initialized: {len(self.clients)}")

            

    # --------------------------------------------------
    # Communication
    # --------------------------------------------------

    def broadcast_model(self):
        """Send current global model to all clients."""
        for c in self.clients:
            c.set_global_model(copy.deepcopy(self.global_model))

    def collect_updates(self, selected_clients, round_id):
        """
        Collect local updates from selected clients.
        """
        updates = []
        max_norm = self.config['federation'].get('max_update_norm', None)

        for c in selected_clients:
            upd = c.local_update(round_id=round_id)

            if upd is None:
                continue  # safety

            if max_norm is not None:
                upd = clip_by_l2(upd, max_norm)
                # Debug: check norm
                total_norm = 0.0
                for k, v in upd.items():
                    total_norm += (v.float() ** 2).sum().item()
                total_norm = total_norm ** 0.5
                print(f"[Server] Client {c.client_id} update norm after clipping: {total_norm:.4f} (max {max_norm})")

            updates.append(upd)  # ✅ append the SAME update

        print(f"[Server] Collected {len(updates)} updates")
        return updates
    # --------------------------------------------------
    # Byzantine client simulation
    # --------------------------------------------------

    def inject_byzantine(self, updates):
        """
        Replace a few updates with extreme malicious updates to test robustness.
        """
        n_byz = self.config['federation'].get('byzantine_clients', 0)
        magnitude = 10.0  # scaling factor for malicious updates
        if n_byz <= 0 or len(updates) == 0:
            return updates

        byz_indices = np.random.choice(len(updates), min(n_byz, len(updates)), replace=False)
        for idx in byz_indices:
            for k in updates[idx].keys():
                updates[idx][k] = updates[idx][k] * magnitude
        print(f"[Server] Injected Byzantine updates at indices {list(byz_indices)}")
        return updates

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------

    def aggregate(self, updates):
        """
        Aggregate client updates using configured strategy.
        """
        method = self.config['federation'].get('aggregation', 'fedavg')
        f = self.config['federation'].get('byzantine_clients', 1)
        print(f"[Server] Aggregating {len(updates)} updates using {method}")

        return aggregate(
            updates,
            method=method,
            f=f
        )

    # --------------------------------------------------
    # Global Model Update
    # --------------------------------------------------

    def update_global_model(self, new_state):
        """Update server global model and broadcast."""
        self.global_model = new_state
        for c in self.clients:
            c.update_server_version(copy.deepcopy(self.global_model))

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def evaluate_global(self):
        """
        Evaluate global model across clients.
        """
        losses = []
        for c in self.clients:
            m = c.evaluate_model(self.global_model)
            losses.append(m.get('loss', 0.0))

        metrics = {
            "avg_loss": sum(losses) / len(losses) if losses else 0.0
        }

        self.logger.log({"event": "evaluate", "metrics": metrics})
        return metrics

    # --------------------------------------------------
    # Federated Training Loop
    # --------------------------------------------------

    def run_rounds(self, num_rounds=5):
        """Main federated training loop."""
        for r in range(num_rounds):
            print(f"[Server] Starting round {r + 1}/{num_rounds}")

            # 1️⃣ Client selection
            selected = random_selection(
                self.clients,
                frac=self.config['federation']['frac']
            )

            # 2️⃣ Broadcast global model
            self.broadcast_model()

            # 3️⃣ Collect updates
            updates = self.collect_updates(selected, round_id=r)
            # 3a️⃣ Inject Byzantine clients
            updates = self.inject_byzantine(updates)
            # 4️⃣ Robust aggregation
            agg_state = self.aggregate(updates)

            # 5️⃣ Update global model
            self.update_global_model(agg_state)

            # 6️⃣ Evaluation
            metrics = self.evaluate_global()
            print(f"[Server] Round {r + 1} metrics: {metrics}")
            print(f"[Server] Selected {len(selected)} clients out of {len(self.clients)}")

