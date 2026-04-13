import json
import numpy as np
import matplotlib.pyplot as plt


def plot_all_metrics(results):
    scenarios = list(results.keys())
    agents = ["DQN", "PPO", "Baseline"]
    metrics = ["cost", "sla_violation", "latency", "utilization"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(len(scenarios))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for j, agent in enumerate(agents):
            means = [results[s][agent][metric]["mean"] for s in scenarios]
            stds = [results[s][agent][metric]["std"] for s in scenarios]

            ax.bar(x + j * width, means, width, yerr=stds, label=agent)

        ax.set_title(metric.upper())
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        #ax.set_xlabel("Scenario")
        ax.set_ylabel(metric)
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout()
    plt.show()


def main():
    with open("results.json", "r") as f:
        results = json.load(f)

    plot_all_metrics(results)


if __name__ == "__main__":
    main()