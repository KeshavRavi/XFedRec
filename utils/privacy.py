# utils/privacy.py
import math

class PrivacyAccountant:
    def __init__(self, n_total_clients, noise_multiplier, target_delta=1e-3):
        self.N = n_total_clients
        self.noise_multiplier = noise_multiplier
        self.delta = target_delta
        self.rounds_played = 0

    def step(self, n_sampled_clients):
        self.rounds_played += 1
        self.q = n_sampled_clients / self.N  # Sampling probability

    def get_epsilon(self):
        """
        Estimates the privacy budget (epsilon) spent so far using Advanced Composition.
        """
        if self.noise_multiplier == 0 or self.rounds_played == 0:
            return 0.0
            
        # Simplified Gaussian Mechanism bounds for FL
        q = self.q
        sigma = self.noise_multiplier
        T = self.rounds_played
        
        # Approximate Epsilon for standard DP-FedAvg
        # Epsilon = sqrt(2 * T * ln(1/delta)) * (q / sigma) + T * q * (e^(q/sigma) - 1)
        term1 = math.sqrt(2 * T * math.log(1 / self.delta)) * (q / sigma)
        term2 = T * q * (math.exp(q / sigma) - 1)
        
        return term1 + term2