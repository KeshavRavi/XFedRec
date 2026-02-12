# drift/detector.py
import math
from drift.adwin import ADWIN2
from drift.kalman import KalmanFilter1D


class DriftDetector:
    def __init__(
        self,
        delta=0.05,                 # IMPORTANT: less strict than 0.002
        kalman_q=1e-5,
        kalman_r=1e-1,
        confidence_threshold=0.7,
        ema_alpha=0.3
    ):
        self.delta = float(delta)
        self.kalman_q = float(kalman_q)
        self.kalman_r = float(kalman_r)
        self.confidence_threshold = float(confidence_threshold)
        self.ema_alpha = float(ema_alpha)


        self.adwins = {}
        self.kalmans = {}
        self.confirmed = set()

        # running stats for normalization (per client)
        self.stats = {}  # client_id -> (mean, M2, n)

        # confidence per client
        self._conf = {}

    def _init_client(self, client_id):
        self.adwins[client_id] = ADWIN2(delta=self.delta)
        self.kalmans[client_id] = KalmanFilter1D(
            process_variance=self.kalman_q,
            measurement_variance=self.kalman_r
        )

    def update(self, client_id, loss_value):
        if client_id not in self.adwins:
            self._init_client(client_id)

        adwin = self.adwins[client_id]
        kalman = self.kalmans[client_id]

        raw_loss = float(loss_value)

        # 1) smooth
        smoothed_loss = float(kalman.update(raw_loss))

        # 2) normalize the SMOOTHED stream (not raw)
        norm_loss = self._norm_loss(client_id, smoothed_loss)

        # 3) ADWIN on normalized stream
        drift_raw = adwin.update(norm_loss)

        # 4) EMA confidence
        prev_conf = self._conf.get(client_id, 0.0)
        a = self.ema_alpha
        new_conf = (1 - a) * prev_conf + a * (1.0 if drift_raw else 0.0)
        self._conf[client_id] = new_conf

        if new_conf >= self.confidence_threshold:
            self.confirmed.add(client_id)

        return {
            "client_id": client_id,
            "loss": raw_loss,
            "smoothed_loss": smoothed_loss,
            "norm_loss": norm_loss,
            "adwin_drift": drift_raw,
            "drift_confidence": float(new_conf),
            "confirmed": client_id in self.confirmed,

            # debug from ADWIN
            "adwin_n": getattr(adwin, "last_n", None),
            "adwin_mean0": getattr(adwin, "last_mean0", None),
            "adwin_mean1": getattr(adwin, "last_mean1", None),
            "adwin_eps": getattr(adwin, "last_eps", None),
        }

    def drift_detected(self, client_id):
        return client_id in self.confirmed

    def reset_client(self, client_id):
        if client_id in self.adwins:
            self.adwins[client_id].reset()
        if client_id in self.kalmans:
            self.kalmans[client_id].reset()

        self.confirmed.discard(client_id)
        self._conf.pop(client_id, None)
        self.stats.pop(client_id, None)

    def has_drifted(self, client_id=None):
        if client_id is None:
            return len(self.confirmed) > 0
        return client_id in self.confirmed

    def _norm_loss(self, client_id, x):
        # Welford running mean/variance
        mean, M2, n = self.stats.get(client_id, (0.0, 0.0, 0))
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        var = (M2 / (n - 1)) if n > 1 else 1e-6
        self.stats[client_id] = (mean, M2, n)

        std = math.sqrt(var) + 1e-6
        z = (x - mean) / std

        # keep it bounded for ADWIN
        # clamp z to avoid exp overflow
        z = max(-3.0, min(3.0, z))
        return (z + 3.0) / 6.0
