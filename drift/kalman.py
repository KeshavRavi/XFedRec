"""
Kalman Filter for smoothing concept drift signals.

Used to reduce false positives from ADWIN2
and provide a continuous drift confidence score.
"""

class KalmanFilter1D:
    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-1,
        initial_estimate: float = 0.0,
        initial_error: float = 1.0
    ):
        """
        Parameters
        ----------
        process_variance : float
            Controls trust in model dynamics (Q)
        measurement_variance : float
            Controls trust in ADWIN signal (R)
        """
        self.Q = float(process_variance)
        self.R = float(measurement_variance)

        self.x = float(initial_estimate)      # State estimate (drift level)
        self.P = float(initial_error)         # Estimation error covariance

    def update(self, measurement: float):
        """
        Update filter with new measurement.

        Parameters
        ----------
        measurement : float
            Drift signal (0 or 1 from ADWIN2)

        Returns
        -------
        float
            Smoothed drift confidence in [0, 1]
        """

        # 1️ Prediction step
        self.P = self.P + self.Q

        # 2️ Kalman gain
        K = self.P / (self.P + self.R)

        # 3️ Update estimate
        self.x = self.x + K * (measurement - self.x)

        # 4️ Update uncertainty
        self.P = (1 - K) * self.P

        return float(self.x)

    def reset(self):
        """Reset filter after confirmed drift adaptation."""
        self.x = 0.0
        self.P = 1.0

    def get_state(self):
        """Return internal state for logging/debugging."""
        return {
            "estimate": self.x,
            "uncertainty": self.P
        }
