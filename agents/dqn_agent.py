# agents/dqn_agent.py

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from envs.cloud_env import CloudEnv
import os


def train_dqn(total_timesteps=100_000, model_path="models/dqn_cloud"):

    os.makedirs("models", exist_ok=True)

    env = Monitor(CloudEnv())

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)

    env.close()

    return model