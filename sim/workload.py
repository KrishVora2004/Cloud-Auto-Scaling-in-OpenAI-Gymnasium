# sim/workload.py

import numpy as np


class WorkloadGenerator:

    def __init__(self, steps, scenario="default", seed=None, sequence=None):
        self.steps = steps
        self.scenario = scenario

        if sequence is not None:
            # 🔥 Use provided sequence (for fair comparison)
            self.sequence = sequence
        else:
            if seed is not None:
                np.random.seed(seed)
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

            if self.scenario == "spike":
                val = 50 if t < 80 else 500

            elif self.scenario == "linear":
                val = 10 + t * 2

            elif self.scenario == "noisy":
                val = np.random.randint(20, 500)

            elif self.scenario == "periodic":
                val = 200 + 150 * np.sin(2 * np.pi * t / 50)

            else:
                base = 100
                periodic = 60 * np.sin(0.05 * t)
                noise = np.random.normal(0, 10)
                burst = np.random.uniform(100, 200) if np.random.rand() < 0.05 else 0
                val = max(0, base + periodic + noise + burst)

            seq.append(val)

        return seq

    def get(self, t):
        if t < len(self.sequence):
            return self.sequence[t]
        return self.sequence[-1]