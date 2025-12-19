# server/server.py
"""
Server orchestration for federated learning.
"""

import copy
import torch
from .aggregator import fedavg, coordinate_median, krum, fedq_placeholder
from .selection import random_selection
from .logger import ExperimentLogger

AGG_MAP = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "krum": krum,
    "fedq": fedq_placeholder
}

class Server:
    def __init__(self, clients, config):
        """
        clients: list of Client objects
        config: experiment config dict
        """
        self.clients = clients
        self.config = config
        self.global_model = None
        self.logger = ExperimentLogger()
        # initialize a global model by copying from first client
        if len(clients) > 0:
            self.global_model = copy.deepcopy(clients[0].get_model_state())
        self.aggregation_fn = AGG_MAP.get(config['aggregation']['method'], fedavg)

    def broadcast_model(self):
        """Send current global model to all clients (server->client)."""
        for c in self.clients:
            c.set_global_model(copy.deepcopy(self.global_model))

    def collect_updates(self, selected_clients):
        """
        Each selected client performs local training and returns a local update (state_dict).
        """
        updates = []
        for c in selected_clients:
            upd = c.local_update()
            updates.append(upd)
        return updates

    def aggregate(self, updates):
        """
        Aggregate using selected aggregator.
        """
        return self.aggregation_fn(updates)

    def update_global_model(self, new_state):
        """Replace server global model with aggregated weights."""
        self.global_model = new_state
        # Optionally distribute biases like state
        for c in self.clients:
            c.update_server_version(copy.deepcopy(self.global_model))

    def evaluate_global(self):
        """
        Basic evaluation calling clients' evaluation routines (can be customized).
        Returns aggregated metrics dict.
        """
        metrics = {}
        losses = []
        for c in self.clients:
            m = c.evaluate_model(self.global_model)
            losses.append(m.get('loss', 0.0))
        metrics['avg_loss'] = sum(losses) / len(losses) if losses else 0.0
        self.logger.log({'event': 'evaluate', 'metrics': metrics})
        return metrics

    def run_rounds(self, num_rounds=5):
        """Main federated loop."""
        for r in range(num_rounds):
            print(f"[Server] Starting round {r+1}/{num_rounds}")
            # select clients
            selected = random_selection(self.clients, frac=self.config['federation']['frac'])
            # broadcast
            self.broadcast_model()
            # collect updates (PASS round_id)
            updates = []
            for c in selected:
                updates.append(c.local_update(round_id=r))
            # aggregate
            agg = self.aggregate(updates)
            # update global
            self.update_global_model(agg)
            # evaluate
            metrics = self.evaluate_global()
            print(f"[Server] Round {r+1} metrics: {metrics}")
