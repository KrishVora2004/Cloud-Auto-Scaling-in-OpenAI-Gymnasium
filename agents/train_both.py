import os
from datetime import datetime

from agents.dqn_agent import train_dqn
from agents.ppo_agent import train_ppo
from sim.workload import WorkloadGenerator


def train_both(total_steps=100000):

    os.makedirs("models", exist_ok=True)

    # SAME base workload 
    base_workload = WorkloadGenerator(total_steps, seed=42)

    # Create independent copies
    workload_dqn = WorkloadGenerator(total_steps, sequence=base_workload.sequence)
    workload_ppo = WorkloadGenerator(total_steps, sequence=base_workload.sequence)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Training DQN...")
    dqn_model = train_dqn(
        total_timesteps=total_steps,
        model_path=f"models/dqn_{timestamp}",
        workload=workload_dqn
    )

    print(f"DQN saved at models/dqn_{timestamp}.zip")

    print("\nTraining PPO...")
    ppo_model = train_ppo(
        total_timesteps=total_steps,
        model_path=f"models/ppo_{timestamp}",
        workload=workload_ppo
    )

    print(f"PPO saved at models/ppo_{timestamp}.zip")

    return dqn_model, ppo_model