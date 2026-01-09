"""
Unified Drift Detector combining ADWIN2 and Kalman Filter.

Tracks concept drift per client and provides
smoothed drift confidence signals.
"""

from drift.adwin import ADWIN2
from drift.kalman import KalmanFilter1D


class DriftDetector:
    def __init__(
        self,
        delta=0.002,
        kalman_q=1e-5,
        kalman_r=1e-1,
        confidence_threshold=0.7
    ):
        """
        Parameters
        ----------
        delta : float
            ADWIN confidence parameter
        kalman_q : float
            Kalman process variance
        kalman_r : float
            Kalman measurement variance
        confidence_threshold : float
            Threshold to trigger drift adaptation
        """
        self.delta = delta
        self.confidence_threshold = confidence_threshold

        self.adwins = {}
        self.kalmans = {}

    def _init_client(self, client_id):
        """Initialize drift detectors for a new client."""
        self.adwins[client_id] = ADWIN2(delta=self.delta)
        self.kalmans[client_id] = KalmanFilter1D(
            process_variance=1e-5,
            measurement_variance=1e-1
        )

    def update(self, client_id, loss_value):
        """
        Update drift detectors with new loss value.

        Returns
        -------
        dict
            Drift monitoring information
        """
        if client_id not in self.adwins:
            self._init_client(client_id)

        adwin = self.adwins[client_id]
        kalman = self.kalmans[client_id]

        # ADWIN raw signal
        drift_raw = adwin.update(loss_value)

        # Convert to numeric signal
        measurement = 1.0 if drift_raw else 0.0

        # Kalman smoothing
        drift_confidence = kalman.update(measurement)

        return {
            "client_id": client_id,
            "loss": loss_value,
            "adwin_drift": drift_raw,
            "drift_confidence": drift_confidence
        }

    def drift_detected(self, client_id):
        """Check if smoothed drift exceeds threshold."""
        if client_id not in self.kalmans:
            return False
        return self.kalmans[client_id].x >= self.confidence_threshold

    def reset_client(self, client_id):
        """Reset drift detectors after adaptation."""
        if client_id in self.adwins:
            self.adwins[client_id].reset()
        if client_id in self.kalmans:
            self.kalmans[client_id].reset()
    def has_drifted(self):
        """
        Returns True if confirmed drift detected
        """
        return self.confirmed
