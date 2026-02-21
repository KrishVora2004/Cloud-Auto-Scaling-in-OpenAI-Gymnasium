# agents/train.py

import argparse
from ppo_agent import train_ppo
from dqn_agent import train_dqn


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn"])
    parser.add_argument("--steps", type=int, default=100_000)

    args = parser.parse_args()

    if args.algo == "ppo":
        train_ppo(total_timesteps=args.steps)

    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.steps)


if __name__ == "__main__":
    main()