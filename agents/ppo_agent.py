# agents/ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.cloud_env import CloudEnv
import os


def train_ppo(total_timesteps=100_000, model_path="models/ppo_cloud"):

    os.makedirs("models", exist_ok=True)

    env = Monitor(CloudEnv())

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)

    env.close()

    return model