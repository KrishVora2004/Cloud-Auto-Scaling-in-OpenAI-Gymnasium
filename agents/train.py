# agents/train.py

import argparse
from agents.ppo_agent import train_ppo
from agents.dqn_agent import train_dqn
from agents.train_both import train_both


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "both"], default="ppo")
    parser.add_argument("--steps", type=int, default=100_000)

    args = parser.parse_args()

    if args.algo == "ppo":
        train_ppo(total_timesteps=args.steps)

    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.steps)

    elif args.algo == "both":
        train_both(total_steps=args.steps)


if __name__ == "__main__":
    main()