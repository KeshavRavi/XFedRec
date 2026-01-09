"""
ADWIN2: Adaptive Windowing Drift Detection

Detects concept drift in a data stream by maintaining
a dynamically sized window and testing for statistically
significant changes in the mean.
"""

import math
from collections import deque


class ADWIN2:
    def __init__(
        self,
        delta: float = 0.002,
        min_window_size: int = 10,
        max_window_size: int = 200
    ):
        """
        Parameters
        ----------
        delta : float
            Confidence level for change detection (smaller = stricter)
        min_window_size : int
            Minimum window size before testing drift
        max_window_size : int
            Maximum window size to avoid memory explosion
        """
        self.delta = delta
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        self.window = deque()
        self.drift_detected = False
        self.last_cut = None

    def reset(self):
        """Reset detector state after confirmed drift."""
        self.window.clear()
        self.drift_detected = False
        self.last_cut = None

    def update(self, value: float):
        """
        Add new observation and test for drift.

        Returns
        -------
        drift : bool
            True if drift detected at this step
        """
        self.window.append(value)

        # Enforce maximum window size
        if len(self.window) > self.max_window_size:
            self.window.popleft()

        self.drift_detected = False

        # Not enough data yet
        if len(self.window) < self.min_window_size:
            return False

        return self._check_drift()

    def _check_drift(self):
        """
        Perform statistical test for drift.
        """
        n = len(self.window)
        values = list(self.window)

        for cut in range(self.min_window_size, n - self.min_window_size):
            w0 = values[:cut]
            w1 = values[cut:]

            mean0 = sum(w0) / len(w0)
            mean1 = sum(w1) / len(w1)

            diff = abs(mean0 - mean1)

            eps = self._compute_epsilon(len(w0), len(w1))

            if diff > eps:
                # Drift detected → shrink window
                self.window = deque(w1)
                self.drift_detected = True
                self.last_cut = cut
                return True

        return False

    def _compute_epsilon(self, n0, n1):
        """
        Compute ADWIN threshold epsilon.
        """
        m = (1 / n0) + (1 / n1)
        return math.sqrt(2 * m * math.log(2 / self.delta))

    def get_state(self):
        """Return current detector state (for logging)."""
        return {
            "window_size": len(self.window),
            "drift_detected": self.drift_detected,
            "last_cut": self.last_cut
        }
