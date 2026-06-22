import os
import glob
from stable_baselines3 import PPO, DQN


class ModelLoadError(Exception):
    """Raised when a requested model file can't be found or fails to load."""
    pass


def _latest_file(pattern):
    """Returns the path of the most recently MODIFIED file matching pattern."""
    matches = glob.glob(pattern)
    if not matches:
        raise ModelLoadError(f"No files found matching pattern: {pattern}")

    return max(matches, key=os.path.getmtime)


def load_latest_models(models_dir="models"):
    """
    Loads the most recently created ppo_*.zip and dqn_*.zip from models_dir.

    Returns: (ppo_model, dqn_model)
    Raises: ModelLoadError if either file is missing or fails to load.
    """
    try:
        ppo_path = _latest_file(os.path.join(models_dir, "ppo_*.zip"))
        dqn_path = _latest_file(os.path.join(models_dir, "dqn_*.zip"))

        ppo_model = PPO.load(ppo_path)
        dqn_model = DQN.load(dqn_path)

        print(f"Loaded latest PPO model: {ppo_path}")
        print(f"Loaded latest DQN model: {dqn_path}")

        return ppo_model, dqn_model

    except Exception:
        print("Failed to load models")
        raise ModelLoadError("Failed to load models") from None


def load_specific_models(ppo_path, dqn_path):
    """
    Loads exact, pinned model files rather than auto-selecting the latest.
    Use this when comparing a specific older checkpoint, reproducing a
    past result, or pinning a model for a report.

    Returns: (ppo_model, dqn_model)
    Raises: ModelLoadError if either file is missing or fails to load.
    """
    try:
        ppo_model = PPO.load(ppo_path)
        dqn_model = DQN.load(dqn_path)

        print(f"Loaded PPO model: {ppo_path}")
        print(f"Loaded DQN model: {dqn_path}")

        return ppo_model, dqn_model

    except Exception:
        print("Failed to load models")
        raise ModelLoadError("Failed to load models") from None