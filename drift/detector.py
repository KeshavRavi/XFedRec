# drift/detector.py
import math
import numpy as np
from drift.adwin import ADWIN2
from drift.kalman import KalmanFilter1D

class DriftDetector:
    def __init__(self, delta=0.01, kalman_q=1e-3, kalman_r=1e-2, confidence_threshold=0.7, ema_alpha=0.3):
        self.delta = delta
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        
        self.adwins = {}
        self.kalmans = {}
        self.stats = {}       
        self.drift_states = {}
        self.baseline_loss = {}

    def _init_client(self, client_id):
        self.adwins[client_id] = ADWIN2(delta=self.delta, min_window_size=2)
        self.kalmans[client_id] = KalmanFilter1D(
            process_variance=self.kalman_q, 
            measurement_variance=self.kalman_r
        )
        self.drift_states[client_id] = "NORMAL"
        self.stats[client_id] = {"mean": 0.0, "var": 0.0, "count": 0}

    def update(self, client_id, loss_value, round_id=None):
        if client_id not in self.adwins:
            self._init_client(client_id)

        # 1. Burn-in
        if round_id is not None and round_id <= 1:
             return self._empty_result(client_id, loss_value)

        kalman = self.kalmans[client_id]
        adwin = self.adwins[client_id]
        
        # 2. Kalman Smoothing
        raw_loss = float(loss_value)
        smoothed_loss = float(kalman.update(raw_loss))

        # 3. Initialization
        if self.stats[client_id]["count"] == 0:
             self.stats[client_id] = {"mean": smoothed_loss, "var": 0.005, "count": 1}

        # 4. Calculate Z-Score
        norm_val, z_score = self._get_norm_val(client_id, smoothed_loss)

        # 5. Hybrid Drift Detection
        drift_detected = False
        
        # A) ADWIN Check (Good for Gradual Drift)
        if adwin.update(norm_val):
            drift_detected = True
            
        # B) Z-Score Check (Good for Sudden Drift)
        # If Z > 3.5 (roughly p < 0.0005), it IS a drift. Don't wait for ADWIN.
        if abs(z_score) > 3.5:
            drift_detected = True
            # Force feed ADWIN so it updates its internal mean for next time
            for _ in range(10): adwin.update(norm_val)

        # DEBUG PRINT
        if round_id is not None and round_id >= 4 and abs(z_score) > 2.0:
             print(f"   [ DETECTOR C{client_id}] Round {round_id} | Loss {raw_loss:.2f} | Z-Score {z_score:.2f} | Drift? {drift_detected}")

        # 6. Freeze Stats Logic
        if self.drift_states[client_id] == "NORMAL":
            # Only update stats if data looks normal
            if abs(z_score) < 3.0: 
                self._update_stats(client_id, smoothed_loss)
            
            prev = self.baseline_loss.get(client_id, raw_loss)
            self.baseline_loss[client_id] = 0.8 * prev + 0.2 * raw_loss

        return {
            "client_id": client_id,
            "raw_loss": raw_loss,
            "smoothed_loss": smoothed_loss,
            "adwin_drift": drift_detected,
            "state": self.drift_states[client_id],
            "baseline": self.baseline_loss.get(client_id, 0.0),
            "z_score": z_score
        }

    def confirm_drift(self, client_id):
        self.drift_states[client_id] = "DRIFTED"
        self.kalmans[client_id].reset()
        
    def signal_recovery(self, client_id):
        self.drift_states[client_id] = "NORMAL"
        self.adwins[client_id].reset()
        self.stats[client_id]["count"] = 0 

    def _update_stats(self, client_id, x):
        s = self.stats[client_id]
        alpha = 0.1 
        diff = x - s["mean"]
        incr = alpha * diff
        s["mean"] = s["mean"] + incr
        s["var"] = (1 - alpha) * (s["var"] + diff * incr)
        self.stats[client_id] = s

    def _get_norm_val(self, client_id, x):
        s = self.stats[client_id]
        if s["var"] == 0: return 0.5, 0.0
        std = math.sqrt(s["var"]) + 1e-9
        z = (x - s["mean"]) / std
        z_clipped = max(-4.0, min(4.0, z))
        norm = (z_clipped + 4.0) / 8.0
        return norm, z

    def _empty_result(self, client_id, loss):
        return {
            "client_id": client_id, "raw_loss": loss, "smoothed_loss": loss,
            "adwin_drift": False, "state": "NORMAL", "baseline": 0.0, "z_score": 0.0
        }