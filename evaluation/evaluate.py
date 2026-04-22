import numpy as np
import glob
from envs.cloud_env import CloudEnv
from stable_baselines3 import PPO, DQN
from .baseline import threshold_policy
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


def evaluate_all(seed=None):

    #  Generate ONE shared workload
    wg = WorkloadGenerator(500, seed=seed)    

    #  SAME workload for all models (fair comparison)
    ppo_env = CloudEnv(WorkloadGenerator(500, sequence=wg.sequence))
    dqn_env = CloudEnv(WorkloadGenerator(500, sequence=wg.sequence))
    baseline_env = CloudEnv(WorkloadGenerator(500, sequence=wg.sequence))

    # Load models manually
    #ppo_model = PPO.load("models/ppo_st_1000")
    #dqn_model = DQN.load("models/dqn_st_1000")

    #load models on the basis of latest created .zip files for each agent
    dqn_path = sorted(glob.glob("models/dqn*.zip"))[-1]
    ppo_path = sorted(glob.glob("models/ppo*.zip"))[-1]
    ppo_model = PPO.load(ppo_path)
    dqn_model = DQN.load(dqn_path)

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
        print(f"\n--- RUN {i+1} ---")
        results.append(evaluate_all(seed=i))  # ✅ different workload each run

    return results


if __name__ == "__main__":

    ppo, dqn, baseline = evaluate_all()

    print("\n--- RESULTS ---")
    print(f"PPO -> Cost: {ppo['cost']:.2f}, Error: {ppo['error']:.3f}, Response: {ppo['response']:.3f}")
    print(f"DQN -> Cost: {dqn['cost']:.2f}, Error: {dqn['error']:.3f}, Response: {dqn['response']:.3f}")
    print(f"BASELINE -> Cost: {baseline['cost']:.2f}, Error: {baseline['error']:.3f}, Response: {baseline['response']:.3f}")

    plot_instances_vs_workload(ppo, dqn, baseline)