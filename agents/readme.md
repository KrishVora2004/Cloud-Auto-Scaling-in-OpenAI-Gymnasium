# Agents

This module contains the core reinforcement learning agents used for cloud resource optimization. It supports multiple algorithms (PPO and DQN) that can be trained interchangeably on the same environment.

The design focuses on modularity, allowing easy experimentation, benchmarking, and extension with new algorithms.

---

## Overview

The `agents` folder includes:

- **Training Pipeline (`train.py`)**  
  Entry point for training RL agents. Handles:
  - Algorithm selection (PPO / DQN)
  - Environment initialization
  - Model training loop
  - Saving trained models

- **Algorithm Implementations**  
  - PPO agent (policy-based, stable for continuous adaptation)
  - DQN agent (value-based, efficient for discrete decisions)

- **Configuration Handling**  
  Centralized control of training parameters such as:
  - Number of steps
  - Learning rate
  - Exploration strategy (for DQN)
  - Policy updates (for PPO)

---

## How to Run (VS Code Terminal)

### Format

```bash
python agents/train.py --algo {algorithm_name} --steps {number_of_steps}
```

### Train DQN

```bash
python agents/train.py --algo dqn --steps 200000
```

### Train PPO

```bash
python agents/train.py --algo ppo --steps 200000
```

---
## Key Features

### Algorithm Flexibility
Easily switch between PPO and DQN using a single command-line argument.

### Consistent Interface
Both agents follow a unified training pipeline, enabling fair comparison.

### Scalable Training
Designed to handle increasing workloads and longer training durations.

### Extensible Design
New RL algorithms can be added with minimal changes.

---

## Models Saved To
```
models/
```


Each trained model is saved with:
- Algorithm name  
- Training steps (optional metadata)  
- Ready for evaluation or deployment  

---

## What This Setup Gives You

- Clean separation of training logic and algorithms  
- PPO and DQN interchangeable  
- Reproducible training runs  
- Easy hyperparameter tuning  
- Easy benchmarking between algorithms  

---

## Notes

- Ensure the environment is properly configured before training  
- Use higher training steps for better convergence  
- PPO is generally more stable for dynamic workloads, while DQN is faster for simpler decision spaces  