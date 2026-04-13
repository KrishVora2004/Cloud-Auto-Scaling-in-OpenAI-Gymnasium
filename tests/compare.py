import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.evaluate_models import evaluate_model
from tests.scenarios import SCENARIOS

from stable_baselines3 import DQN, PPO
from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator


# Load latest models automatically
dqn_path = sorted(glob.glob("models/dqn*.zip"))[-1]
ppo_path = sorted(glob.glob("models/ppo*.zip"))[-1]

dqn_model = DQN.load(dqn_path)
ppo_model = PPO.load(ppo_path)


class BaselineAgent:
    def predict(self, obs, deterministic=True):
        utilization = obs[2]

        if utilization > 0.8:
            return 2, None
        elif utilization < 0.3:
            return 0, None
        return 1, None


def env_fn(scenario, seed):
    workload = WorkloadGenerator(steps=500, scenario=scenario, seed=seed)
    return CloudEnv(workload)


def run_comparison():
    results = {}

    baseline = BaselineAgent()

    for scenario in SCENARIOS:
        print(f"\nRunning scenario: {scenario}")

        results[scenario] = {
            "DQN": evaluate_model(env_fn, dqn_model, scenario),
            "PPO": evaluate_model(env_fn, ppo_model, scenario),
            "Baseline": evaluate_model(env_fn, baseline, scenario),
        }

    return results