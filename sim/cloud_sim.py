# sim/cloud_sim.py

import numpy as np


class CloudSimulator:

    # sim/cloud_sim.py
    
    def __init__(self,
                 mu=50,
                 cost_per_instance=1,
                 lambda_base=100,
                 noise_std=10,
                 N_min=1,
                 N_max=20):

        self.mu = mu
        self.cost_per_instance = cost_per_instance
        self.lambda_base = lambda_base
        self.noise_std = noise_std
        self.N_min = N_min
        self.N_max = N_max

        # Initialize base variables so step() doesn't fail
        self.lambda_t = self.lambda_base
        self.N_t = 5
        self.t = 0
        
        # Now it is safe to call reset
        self.reset()

    def reset(self):
        self.lambda_t = self.lambda_base
        self.N_t = 5
        self.t = 0

        return {
            "lambda": self.lambda_t,
            "instances": self.N_t
        }
    

    # -----------------------------------
    def step(self, action, lambda_t):

        self.lambda_t = lambda_t
        
        # Apply scaling action
        if action == 0:
            self.N_t -= 1
        elif action == 2:
            self.N_t += 1

        self.N_t = np.clip(self.N_t, self.N_min, self.N_max)


        capacity = self.N_t * self.mu
        utilization = self.lambda_t / capacity if capacity > 0 else 1.0

        # Queue-based response time
        if self.lambda_t < capacity:
            response_time = 1 / (self.mu - (self.lambda_t / self.N_t))
            error_rate = 0
        else:
            response_time = 10
            error_rate = (self.lambda_t - capacity) / self.lambda_t

        cost = self.N_t * self.cost_per_instance

        self.t += 1

        return {
            "lambda": self.lambda_t,
            "instances": self.N_t,
            "utilization": utilization,
            "response_time": response_time,
            "error_rate": error_rate,
            "cost": cost
        }
