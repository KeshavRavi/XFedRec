# server/logger.py
"""
Simple logging utility for experiments.
"""

import os
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, out_dir="experiments/results"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.filepath = os.path.join(self.out_dir, f"run_{self.run_id}.json")
        self.records = []

    def log(self, record: dict):
        self.records.append(record)
        # Optionally flush to disk
        with open(self.filepath, "w") as f:
            json.dump(self.records, f, indent=2)
