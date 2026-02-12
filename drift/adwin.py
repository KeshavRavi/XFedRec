# drift/adwin.py
import math
from collections import deque


class ADWIN2:
    def __init__(self, delta: float = 0.05, min_window_size: int = 5, max_window_size: int = 200):
        self.delta = delta
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        self.window = deque()
        self.drift_detected = False
        self.last_cut = None

        # debug
        self.last_n = None
        self.last_mean0 = None
        self.last_mean1 = None
        self.last_eps = None

    def reset(self):
        self.window.clear()
        self.drift_detected = False
        self.last_cut = None
        self.last_n = None
        self.last_mean0 = None
        self.last_mean1 = None
        self.last_eps = None

    def update(self, value: float):
        self.window.append(float(value))

        if len(self.window) > self.max_window_size:
            self.window.popleft()

        self.drift_detected = False

        if len(self.window) < self.min_window_size:
            return False

        return self._check_drift()

    def _check_drift(self):
        n = len(self.window)
        values = list(self.window)
        best = None  # (diff, mean0, mean1, eps, cut)

        self.last_n = n
        self.last_mean0 = None
        self.last_mean1 = None
        self.last_eps = None
        self.last_cut = None

        for cut in range(1, n):
            n0 = cut
            n1 = n - cut
            if n0 < self.min_window_size or n1 < self.min_window_size:
                continue

            w0 = values[:cut]
            w1 = values[cut:]

            mean0 = sum(w0) / n0
            mean1 = sum(w1) / n1
            diff = abs(mean0 - mean1)

            eps = self._compute_epsilon(n0, n1, R=1.0)
            eps = min(eps, 1.0)

            if best is None or diff > best[0]:
                best = (diff, mean0, mean1, eps, cut)

            if diff > eps:
                self.window = deque(w1)
                self.drift_detected = True
                self.last_cut = cut
                self.last_mean0 = mean0
                self.last_mean1 = mean1
                self.last_eps = eps
                return True

        if best:
            self.last_mean0 = best[1]
            self.last_mean1 = best[2]
            self.last_eps = best[3]
            self.last_cut = best[4]

        return False


    def _compute_epsilon(self, n0, n1, R=1.0):
        # less conservative than current one
        m = (1.0 / n0) + (1.0 / n1)
        return R * math.sqrt(m * math.log(2.0 / self.delta))

