"""
Server orchestration for federated learning.
"""

import copy
import numpy as np
from .aggregator import aggregate
from .robust_aggregation import fedq_aggregate
from .selection import random_selection
from .logger import ExperimentLogger
from utils.norms import clip_by_l2
from drift.detector import DriftDetector


class Server:
    def __init__(self, clients, config):
        """
        clients: list of Client objects
        config: experiment config dict
        """
        self.clients = clients
        self.config = config
        self.logger = ExperimentLogger()
        self.last_train_losses = {}


        # Initialize global model from first client
        self.global_model = None
        if len(clients) > 0:
            self.global_model = copy.deepcopy(clients[0].get_model_state())
        print(f"[Server] Total clients initialized: {len(self.clients)}")
        drift_cfg = self.config.get("drift", {})
        print(type(drift_cfg.get("kalman_q")), drift_cfg.get("kalman_q"))

        self.drift_detector = DriftDetector(
            delta=drift_cfg.get("delta", 0.05),
            kalman_q=drift_cfg.get("kalman_q", 1e-5),
            kalman_r=drift_cfg.get("kalman_r", 1e-1),
            confidence_threshold=drift_cfg.get("confidence_threshold", 0.6),
            ema_alpha=drift_cfg.get("ema_alpha", 0.3)
        )
        self.drifted_clients = set()
            

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
        client_ids = []
        qualities = []  #  FedQ quality scores
        max_norm = self.config['federation'].get('max_update_norm', None)

        for c in selected_clients:
            upd,metrics = c.local_update(round_id=round_id)
            if upd is None:
                print(f"[Server] Client {c.client_id} returned upd=None")
                continue
            if not isinstance(metrics, dict):
                print(f"[Server] Client {c.client_id} metrics is {type(metrics)}, using default quality")
                metrics = {}

            if not isinstance(upd, dict):
                raise TypeError(f"[Server] Client {c.client_id} update is {type(upd)} not dict. Did you return (state, metrics) but not unpack?")
            
            self.last_train_losses[c.client_id] = metrics.get("train_loss", None)
            # Quality = inverse of training loss
            train_loss = metrics.get("train_loss", None)
            if train_loss is None:
                q=1.0
            else:
                q = 1.0 / (train_loss + 1e-6)
            qualities.append(q)

            if max_norm is not None:
                upd = clip_by_l2(upd, max_norm)
                # Debug: check norm
                total_norm = 0.0
                for k, v in upd.items():
                    total_norm += (v.float() ** 2).sum().item()
                total_norm = total_norm ** 0.5
                print(f"[Server] Client {c.client_id} update norm after clipping: {total_norm:.4f} (max {max_norm})")

            updates.append(upd)  # append the SAME update
            client_ids.append(c.client_id)
            
        self.qualities = qualities    
        print(f"[Server] Collected {len(updates)} updates")
        return updates,client_ids
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

    def aggregate(self, updates, client_ids):
        """
        Aggregate client updates using robust aggregation,
        excluding drifted clients when possible.
        """
        method = self.config['federation'].get('aggregation', 'fedavg')
        f = self.config['federation'].get('byzantine_clients', 1)
        
        self.drifted_clients = getattr(self, "drifted_clients", set())

        filtered_updates = []
        filtered_client_ids = []

        for upd, cid in zip(updates, client_ids):
            if cid in self.drifted_clients:
                print(f"[Server] Excluding Client {cid} due to confirmed drift")
                continue
            filtered_updates.append(upd)
            filtered_client_ids.append(cid)

        # Fallback if all clients drifted
        if not filtered_updates:
            print("[Server] All clients drifted — using all updates")
            filtered_updates = updates
            filtered_client_ids = client_ids
        if not filtered_updates:
            raise RuntimeError("No updates available for aggregation")

        print(
            f"[Server] Aggregating {len(filtered_updates)} updates "
            f"using {method}"
        )

        if method == "fedq":
            # Use qualities collected during collect_updates()
            qualities = getattr(self, "qualities", None)
            if qualities is None or len(qualities) != len(updates):
                    qualities = [1.0] * len(updates)

            # Filter qualities alongside updates
            filtered_qualities = []
            for cid, q in zip(client_ids, qualities):
                if cid in self.drifted_clients:
                    continue
                filtered_qualities.append(q)
            # Safety fallback
            if len(filtered_qualities) != len(filtered_updates):
                filtered_qualities = [1.0] * len(filtered_updates)

            return fedq_aggregate(filtered_updates, filtered_qualities)
        
        # IMPORTANT: for non-fedq methods, return normal aggregator result
        return aggregate(filtered_updates, method=method, f=f)
    # --------------------------------------------------
    # Global Model Update
    # --------------------------------------------------

    def update_global_model(self, new_state):
        """Update server global model and broadcast."""
        if new_state is None:
            raise RuntimeError("[Server] Aggregation returned None (new_state is None). Check aggregator output.")

        if not isinstance(new_state, dict):
            raise TypeError(f"[Server] Aggregation returned {type(new_state)} instead of dict")

        self.global_model = new_state

        for c in self.clients:
            c.update_server_version(copy.deepcopy(self.global_model))


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def evaluate_global(self, current_round):
        """
        Evaluate, Detect Drift, and Manage Adaptation Lifecycle.
        """
        losses = []
        drift_metrics_log = []

        # Ground Truth Config for Metric Calculation
        drift_cfg = self.config.get("drift", {})
        gt_drift_round = drift_cfg.get("round", -1)
        gt_enabled = drift_cfg.get("enabled", False)

        for c in self.clients:
            # 1. Get Loss (Evaluation)
            metrics = c.evaluate_model(self.global_model)
            # Use training loss if available for smoother signal, else eval loss
            train_loss = self.last_train_losses.get(c.client_id)
            loss = train_loss if train_loss is not None else metrics.get("loss", 0.0)
            losses.append(loss)

            # 2. Feed into Detector
            d_info = self.drift_detector.update(c.client_id, loss, current_round)
            
            # --- LOGIC GATE: Handle Drift ---
            
            # Case A: New Drift Detected
            if d_info["adwin_drift"] and d_info["state"] == "NORMAL":
                print(f"[Drift]  DETECTED for Client {c.client_id} at Round {current_round}")
                
                # 1. Trigger Client Adaptation
                c.adapt_to_drift() 
                
                # 2. Update Server State
                self.drift_detector.confirm_drift(c.client_id)
                self.drifted_clients.add(c.client_id)
                
                # 3. Calculate Detection Delay (Metric)
                if gt_enabled and gt_drift_round > 0:
                    delay = current_round - gt_drift_round
                    print(f"   -> Detection Delay: {delay} rounds")
                    drift_metrics_log.append({
                        "client_id": c.client_id,
                        "metric": "detection_delay",
                        "value": delay,
                        "round": current_round
                    })

            # Case B: Monitoring Recovery
            elif c.client_id in self.drifted_clients:
                # Check if loss has recovered to near baseline (e.g., within 20%)
                baseline = d_info["baseline"]
                if loss <= baseline * 1.2: 
                    print(f"[Drift]  Client {c.client_id} RECOVERED at Round {current_round}")
                    
                    self.drifted_clients.discard(c.client_id)
                    self.drift_detector.signal_recovery(c.client_id)
                    
                    # Calculate Recovery Time (Metric)
                    # Note: You might want to store 'drift_start_round' in a dict to be precise
                    drift_metrics_log.append({
                        "client_id": c.client_id,
                        "metric": "recovery_event",
                        "value": 1, 
                        "round": current_round
                    })

            # Log granular status
            print(f"[Client {c.client_id}] Loss: {loss:.4f} | Smooth: {d_info['smoothed_loss']:.4f} | State: {d_info['state']}")

        # Aggregated Metrics
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Save to logger
        log_payload = {
            "avg_loss": avg_loss,
            "drift_events": drift_metrics_log
        }
        self.logger.log({"event": "evaluate", "metrics": log_payload})
        
        return {"avg_loss": avg_loss, "drift_metrics": drift_metrics_log}

    # --------------------------------------------------
    # Federated Training Loop
    # --------------------------------------------------

    def run_rounds(self, num_rounds=5):
        """Main federated training loop with metric logging."""
        all_metrics = []

        for r in range(num_rounds):
            print(f"[Server] Starting round {r + 1}/{num_rounds}")
            
            #  clear per-round train losses so we never reuse old values
            self.last_train_losses = {}
            # 1️ Client selection
            selected = random_selection(
                self.clients,
                frac=self.config['federation']['frac']
            )

            # 2️ Broadcast global model
            self.broadcast_model()

            # 3️ Collect updates
            updates, client_ids = self.collect_updates(selected, round_id=r)

            # 3a️ Inject Byzantine clients (if enabled)
            updates = self.inject_byzantine(updates)

            # 4️ Robust aggregation
            agg_state = self.aggregate(updates, client_ids)

            # 5️ Update global model
            self.update_global_model(agg_state)

            # 6️ Evaluation
            metrics = self.evaluate_global(current_round=r + 1)
            metrics["round"] = r + 1
            all_metrics.append(metrics)

            print(f"[Server] Round {r + 1} metrics: {metrics}")
            print(f"[Server] Selected {len(selected)} clients out of {len(self.clients)}")

        return all_metrics


