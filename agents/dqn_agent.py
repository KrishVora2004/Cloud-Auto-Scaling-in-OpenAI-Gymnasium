import os
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from envs.cloud_env import CloudEnv
from sim.workload import WorkloadGenerator


def train_dqn(total_timesteps=100_000, model_path=None,
              workload=None, workload_factory=None):

    os.makedirs("models", exist_ok=True)

    if model_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/dqn_{total_timesteps}_{timestamp}"

    if workload_factory is not None:
        env = Monitor(CloudEnv(workload_factory=workload_factory))
    else:
        if workload is None:
            workload = WorkloadGenerator(steps=1000)
        env = Monitor(CloudEnv(workload=workload))

    learning_starts = min(5_000, max(200, total_timesteps // 20))
    buffer_size = min(100_000, max(2_000, total_timesteps))

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)

    env.close()

    print(f"DQN model saved to {model_path}.zip")

    return model