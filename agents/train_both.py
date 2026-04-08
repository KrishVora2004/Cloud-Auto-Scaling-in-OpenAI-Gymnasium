# agents/train_both.py

import os
from datetime import datetime
from stable_baselines3 import PPO, DQN
from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator


def train_both(total_steps=100000):

    os.makedirs("models", exist_ok=True)

    # Generate shared workload
    workload = WorkloadGenerator(total_steps)

    print("Training DQN...")
    env_dqn = CloudEnv(workload)

    dqn_model = DQN("MlpPolicy", env_dqn, verbose=1)
    dqn_model.learn(total_timesteps=total_steps)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dqn_path = f"models/dqn_{timestamp}.zip"
    dqn_model.save(dqn_path)

    print(f"DQN saved at {dqn_path}")

    print("\nTraining PPO...")
    env_ppo = CloudEnv(workload)

    ppo_model = PPO("MlpPolicy", env_ppo, verbose=1)
    ppo_model.learn(total_timesteps=total_steps)

    ppo_path = f"models/ppo_{timestamp}.zip"
    ppo_model.save(ppo_path)

    print(f"PPO saved at {ppo_path}")