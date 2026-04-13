# evaluation/evaluate.py

import numpy as np
from envs.cloud_env import CloudEnv
from stable_baselines3 import PPO, DQN
from .baseline import threshold_policy
import matplotlib.pyplot as plt
from sim.workload import WorkloadGenerator
from .plot_results import plot_instances_vs_workload


def run_episode(env, policy_fn=None, model=None):

    state, _ = env.reset()

    done = False

    total_cost = 0
    total_error = 0
    total_response = 0

    instances_history = []
    workload_history = []

    steps = 0

    while not done:

        if model:
            action, _ = model.predict(state, deterministic=True)
        else:
            action = policy_fn(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        instances = info["instances"]
        utilization = info["utilization"]
        error_rate = info["error_rate"]
        response_time = info["response_time"]

        cost = info["cost"]

        total_cost += cost
        total_error += error_rate
        total_response += response_time

        instances_history.append(instances)
        workload_history.append(next_state[0])

        state = next_state
        steps += 1

        done = terminated or truncated

    return {
        "cost": total_cost / steps,
        "error": total_error / steps,
        "response": total_response / steps,
        "instances": instances_history,
        "workload": workload_history
    }


def evaluate_all():

    ppo_env = CloudEnv(WorkloadGenerator(500, seed=1))
    dqn_env = CloudEnv(WorkloadGenerator(500, seed=2))
    baseline_env = CloudEnv(WorkloadGenerator(500, seed=3))

    # Load models
    # Note: Ensure these paths are to be changed according to models names
    ppo_model = PPO.load("models/ppo_st_1000")
    dqn_model = DQN.load("models/dqn_st_1000")

    print("Running PPO...")
    ppo_results = run_episode(ppo_env, model=ppo_model)

    print("Running DQN...")
    dqn_results = run_episode(dqn_env, model=dqn_model)

    print("Running Baseline...")
    baseline_results = run_episode(baseline_env, policy_fn=threshold_policy)

    return ppo_results, dqn_results, baseline_results

def evaluate_multiple(runs=5):
    results = []
    for i in range(runs):
        results.append(evaluate_all())
    return results


if __name__ == "__main__":

    ppo, dqn, baseline = evaluate_all()

    print("\n--- RESULTS ---")
    print(f"PPO -> Cost: {ppo['cost']:.2f}, Error: {ppo['error']:.3f}, Response: {ppo['response']:.3f}")
    print(f"DQN -> Cost: {dqn['cost']:.2f}, Error: {dqn['error']:.3f}, Response: {dqn['response']:.3f}")
    print(f"BASELINE -> Cost: {baseline['cost']:.2f}, Error: {baseline['error']:.3f}, Response: {baseline['response']:.3f}")
    
    plot_instances_vs_workload(ppo, dqn, baseline)