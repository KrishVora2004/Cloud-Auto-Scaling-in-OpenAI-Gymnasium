import os
import argparse
import numpy as np

from agents.ppo_agent import train_ppo
from agents.dqn_agent import train_dqn
from sim.workload import WorkloadGenerator


def train_both(total_steps=100_000, episode_length=500):

    os.makedirs("models", exist_ok=True)

    scenarios = ["default", "spike", "linear", "noisy", "periodic"]

    weights = [0.40, 0.15, 0.15, 0.15, 0.15]

    rng = np.random.default_rng(42)

    def workload_factory():
        scenario = rng.choice(scenarios, p=weights)
        return WorkloadGenerator(steps=episode_length, scenario=scenario)

    print("Training DQN...")
    dqn_model = train_dqn(
        total_timesteps=total_steps,
        workload_factory=workload_factory,
    )

    print("\nTraining PPO...")
    ppo_model = train_ppo(
        total_timesteps=total_steps,
        workload_factory=workload_factory,
    )

    return dqn_model, ppo_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "both"], default="ppo")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--episode-length", type=int, default=500,
                         help="Workload length per episode (only used by --algo both; "
                              "ppo/dqn alone still build their own default workload)")

    args = parser.parse_args()

    print(f"Starting training: algo={args.algo}, steps={args.steps}\n")

    if args.algo == "ppo":
        train_ppo(total_timesteps=args.steps)

    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.steps)

    elif args.algo == "both":
        train_both(total_steps=args.steps, episode_length=args.episode_length)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()