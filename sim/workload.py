# sim/workload.py

import numpy as np


class WorkloadGenerator:

    def __init__(self, steps):
        self.steps = steps
        self.sequence = self._generate_sequence()

    def _generate_sequence(self):
        """
        Realistic workload model:
        - periodic pattern
        - noise
        - occasional bursts
        """

        seq = []

        for t in range(self.steps):

            base = 100
            periodic = 60 * np.sin(0.05 * t)
            noise = np.random.normal(0, 10)

            burst = 0
            if np.random.rand() < 0.05:
                burst = np.random.uniform(100, 200)

            lambda_t = max(0, base + periodic + noise + burst)
            seq.append(lambda_t)

        return seq

    def get(self, t):
        if t < len(self.sequence):
            return self.sequence[t]
        return self.sequence[-1]