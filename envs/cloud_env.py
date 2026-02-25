# envs/cloud_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim.cloud_sim import CloudSimulator


class CloudEnv(gym.Env):

    def __init__(self):

        super(CloudEnv, self).__init__()

        self.sim = CloudSimulator()

        self.action_space = spaces.Discrete(3)

        low = np.array([0, 1, 0, 0, 0], dtype=np.float32)
        high = np.array([1000, 20, 2, 1, 1], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.max_steps = 500
        self.current_step = 0

    # -----------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Good practice for Gymnasium
        
        metrics = self.sim.reset() # This now returns the full dict from self.step(1)
        self.current_step = 0

        state = self._build_state(metrics)
        return state, {}

    # -----------------------------------
    def step(self, action):

        metrics = self.sim.step(action)

        state = self._build_state(metrics)

        reward = self.compute_reward(metrics, action)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return state, reward, terminated, truncated, {}

    # -----------------------------------
    def _build_state(self, metrics):

        response_norm = min(metrics["response_time"] / 10, 1)

        return np.array([
            metrics["lambda"],
            metrics["instances"],
            metrics["utilization"],
            metrics["error_rate"],
            response_norm
        ], dtype=np.float32)

    # -----------------------------------
    def compute_reward(self, metrics, action):

        max_response = 10
        max_cost = 20

        norm_response = metrics["response_time"] / max_response
        norm_error = metrics["error_rate"]
        norm_cost = metrics["cost"] / max_cost

        alpha = 0.4
        beta = 0.5
        gamma = 0.1

        reward = - (alpha * norm_response +
                    beta * norm_error +
                    gamma * norm_cost)

        # Stability penalty
        scaling_penalty = 0.02 * abs(action - 1)
        reward -= scaling_penalty

        return reward
