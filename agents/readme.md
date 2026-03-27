
# Agents

## How to Run (VS Code Terminal)

### Format

```bash
python agents/train.py --algo {algorithm_name} --steps {number of steps}
```

### Train PPO

```bash
python agents/train.py --algo ppo --steps 200000
```

### Train DQN

```bash
python agents/train.py --algo dqn --steps 200000
```

## Models Saved To

```
models/
```

## What This Setup Gives You

-  Clean separation
-  PPO and DQN interchangeable
-  Reproducible training
-  Easy hyperparameter tuning
-  Easy benchmarking
