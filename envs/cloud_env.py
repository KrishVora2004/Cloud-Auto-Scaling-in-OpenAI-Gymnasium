import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim.cloud_sim import CloudSimulator


class CloudEnv(gym.Env):

    LAMBDA_MAX = 1000.0
    RESPONSE_MAX = 10.0

    def __init__(self, workload=None, workload_factory=None,
                 mu=50, cost_per_instance=1, N_min=1, N_max=20):
        super().__init__()

        if workload is None and workload_factory is None:
            raise ValueError("CloudEnv requires either workload or workload_factory")

        self.sim = CloudSimulator(mu=mu, cost_per_instance=cost_per_instance,
                                   N_min=N_min, N_max=N_max)

        self.workload_factory = workload_factory
        self.workload = workload if workload_factory is None else workload_factory()
        self.t = 0

        self.action_space = spaces.Discrete(3)

        low = np.zeros(5, dtype=np.float32)
        high = np.ones(5, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        metrics = self.sim.reset()
        self.t = 0

        if self.workload_factory is not None:
            self.workload = self.workload_factory()

        metrics["lambda"] = self.workload.get(self.t)

        state = self._build_state(metrics)

        return state, {}

    def step(self, action):
        lambda_t = self.workload.get(self.t)

        metrics = self.sim.step(action, lambda_t)

        state = self._build_state(metrics)
        reward = self.compute_reward(metrics, action)

        self.t += 1

        terminated = False
        truncated = self.t >= len(self.workload.sequence)

        info = {
            "requests": metrics["lambda"],
            "served_requests": metrics["lambda"] * (1 - metrics["error_rate"]),
            "utilization": min(metrics["utilization"], 1.0),
            "response_time": metrics["response_time"],
            "error_rate": metrics["error_rate"],
            "cost": metrics["cost"],
            "instances": metrics["instances"],
            "action": action,
            "reward": reward,
        }

        return state, reward, terminated, truncated, info

    def _build_state(self, metrics):
        """Normalize all 5 features to [0,1] to avoid scale imbalance."""

        lambda_norm = min(metrics["lambda"] / self.LAMBDA_MAX, 1.0)
        N_norm = (metrics["instances"] - self.sim.N_min) / (self.sim.N_max - self.sim.N_min)
        utilization_norm = min(metrics["utilization"], 1.0)
        error_norm = metrics["error_rate"]
        response_norm = min(metrics["response_time"] / self.RESPONSE_MAX, 1.0)

        return np.array([
            lambda_norm,
            N_norm,
            utilization_norm,
            error_norm,
            response_norm,
        ], dtype=np.float32)

    def compute_reward(self, metrics, action):
        

        max_response = self.RESPONSE_MAX
        max_cost = self.sim.N_max * self.sim.cost_per_instance

        norm_response = metrics["response_time"] / max_response
        norm_error = metrics["error_rate"]
        norm_cost = metrics["cost"] / max_cost

        required_instances = np.ceil(metrics["lambda"] / self.sim.mu)
        required_instances = np.clip(
            required_instances, self.sim.N_min, self.sim.N_max
        )
        idle_instances = max(0.0, metrics["instances"] - required_instances)
        norm_idle = idle_instances / self.sim.N_max

        alpha = 0.3
        beta  = 0.6
        gamma = 0.1
        delta = 0.3

        reward = -(
            alpha * norm_response
            + beta  * norm_error
            + gamma * norm_cost
            + delta * norm_idle
        )

        scaling_penalty = 0.02 * abs(action - 1)
        reward -= scaling_penalty

        return reward