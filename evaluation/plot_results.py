# evaluation/plot_results.py

import matplotlib.pyplot as plt


def plot_instances_vs_workload(ppo_results, dqn_results, baseline_results):

    # Shared workload
    workload = ppo_results["workload"]

    plt.figure(figsize=(12, 6))

    # Workload (single line)
    plt.plot(workload, label="Workload (λ)", linestyle="--", linewidth=2)

    # PPO
    plt.plot(ppo_results["instances"], label="PPO", linewidth=2)

    # DQN
    plt.plot(dqn_results["instances"], label="DQN", linewidth=2)

    # Baseline
    plt.plot(baseline_results["instances"], label="Baseline", linewidth=2)

    plt.title("Workload vs Scaling Decisions")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")

    plt.legend()
    plt.grid()

    plt.show()