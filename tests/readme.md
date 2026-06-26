# 🧪 Tests Module (`tests/`)

The **Tests Module** is the experiment orchestration layer of the project. It decides:

* **What** to compare
* **Which scenarios** to evaluate
* **How many seeds** to run
* **How results are aggregated**
* **How results are visualized**

Unlike `evaluation/`, which only provides reusable evaluation mechanics, this module designs and executes the actual experiments.

---

## 📖 Table of Contents

* [Purpose](#-purpose)
* [Role in the Project Pipeline](#-role-in-the-project-pipeline)
* [Module Structure](#-module-structure)
* [Experiment Workflow](#-experiment-workflow)
* [File Descriptions](#-file-descriptions)
* [CLI Commands](#-cli-commands)
* [Summary](#-summary)

---

# 🎯 Purpose

The `tests/` module is responsible for:

✅ Loading trained models

✅ Running multi-scenario experiments

✅ Running multi-seed evaluations

✅ Aggregating metrics

✅ Producing plots and dashboards

✅ Diagnosing reward-function issues

It represents the **final stage** of the RL pipeline.

---

# 🏗️ Role in the Project Pipeline

```text id="fh2v7h"
agents/train.py
        │
        ▼
    models/
        │
        ▼
tests/run_experiments.py
        │
        ├── Dashboard
        │
        └── Metrics
        │
        ▼
tests/run_single_episode.py
        │
        └── Time-Series Plot
        │
        ▼
tests/diagnose_reward.py
        │
        └── Reward Analysis
```

The module consumes trained models and produces all results used to evaluate agent performance.

---

# 📂 Module Structure

| File                    | Responsibility                   |
| ----------------------- | -------------------------------- |
| `scenarios.py`          | Defines evaluation scenarios     |
| `metrics.py`            | Computes and aggregates metrics  |
| `compare.py`            | Runs multi-agent comparisons     |
| `plot_dashboard.py`     | Creates dashboard visualizations |
| `diagnose_reward.py`    | Investigates reward calibration  |
| `run_experiments.py`    | Full experiment runner           |
| `run_single_episode.py` | Single-episode visualization     |

---

# 🔄 Experiment Workflow

```text id="mx3tqm"
Load Models
      │
      ▼
Select Scenario
      │
      ▼
Run Agents
      │
      ▼
Collect Metrics
      │
      ▼
Aggregate Results
      │
      ▼
Generate Plots
```

Every experiment follows this same pipeline.

---

# 📁 File Descriptions

---

# 🌊 `scenarios.py`

Provides a single source of truth for evaluation scenarios.

```python id="48igdl"
SCENARIOS = [
    "spike",
    "linear",
    "noisy",
    "periodic"
]
```

---

## Why Centralize Scenario Names?

Without this file:

❌ Scenario names become hardcoded.

❌ Typos become difficult to detect.

❌ Adding new scenarios requires modifying many files.

By defining them once, every experiment uses the same set of workload scenarios.

---

# 📊 `metrics.py`

Responsible for metric computation and aggregation.

Provides two functions:

---

## `compute_metrics(data)`

Input:

```python id="zwscvd"
list[info_dict]
```

Output:

```python id="yl8ht2"
{
    "cost": ...,
    "error": ...,
    "response": ...,
    "utilization": ...
}
```

---

## Metric Definitions

| Metric        | Aggregation |
| ------------- | ----------- |
| Cost          | Summed      |
| Error Rate    | Mean        |
| Response Time | Mean        |
| Utilization   | Mean        |

---

## `aggregate(results)`

Input:

```python id="vztvtt"
[
    metric_dict_seed1,
    metric_dict_seed2,
    ...
]
```

Output:

```python id="sqdzrw"
{
    "mean": ...,
    "std": ...
}
```

Used to generate:

* Dashboard values
* Error bars
* Final comparison tables

---

# ⚔️ `compare.py`

The central comparison engine.

Provides:

```python id="mdjlwm"
run_comparison()
```

---

## Responsibilities

1. Load latest PPO model
2. Load latest DQN model
3. Create baseline agent
4. Run all scenarios
5. Run all seeds
6. Aggregate metrics
7. Return results

---

## Data Flow

```text id="mx8vnf"
PPO
DQN
Baseline
      │
      ▼
All Scenarios
      │
      ▼
All Seeds
      │
      ▼
Aggregated Results
```

---

## Return Structure

```python id="jlwmqe"
results[
    scenario
][
    agent
][
    metric
]
```

Example:

```python id="ccjlwm"
results["spike"]["ppo"]["cost"]
```

---

# 📈 `plot_dashboard.py`

Responsible for creating the project's primary evaluation figure.

Produces:

```text id="g5s5od"
2 × 2 Dashboard
```

---

## Dashboard Layout

| Subplot      | Metric        |
| ------------ | ------------- |
| Top Left     | Cost          |
| Top Right    | SLA Violation |
| Bottom Left  | Latency       |
| Bottom Right | Utilization   |

---

## Plot Structure

```text id="9qk3po"
Scenario
     │
     ├── PPO
     ├── DQN
     └── Baseline
```

Each bar includes:

```text id="rfjlwm"
Mean ± Standard Deviation
```

computed across evaluation seeds.

---

## Data Source

The dashboard consumes:

```python id="ujtrsp"
results
```

directly in memory.

No intermediate file is required.

---

# 🔍 `diagnose_reward.py`

A diagnostic tool used during reward tuning.

Its purpose is to determine whether poor behavior originates from:

* Insufficient training
* Poor exploration
* Reward miscalibration

---

## Method

Runs:

```text id="qadkkr"
Learned Policy
```

versus

```text id="jpz3nt"
Forced Scale-Up Policy
```

on the exact same workload.

---

## Metrics Compared

* Mean reward
* Mean utilization
* Mean instance count

---

## Interpretation

### Forced Policy Performs Better

```text id="4a4w2x"
Training Problem
```

Possible causes:

* Insufficient timesteps
* Poor exploration
* Hyperparameter issues

---

### Forced Policy Performs Worse

```text id="yfzjlwm"
Reward Problem
```

Possible causes:

* Cost penalties too strong
* SLA penalties too weak
* Idle penalties improperly weighted

---

# 🚀 CLI Commands

---

# Full Multi-Scenario Comparison

```bash id="siyhho"
python -m tests.run_experiments
```

---

## What It Does

1. Load latest PPO model
2. Load latest DQN model
3. Evaluate:

```text id="uv9tbp"
PPO
DQN
Baseline
```

4. Run all scenarios
5. Run three seeds each
6. Aggregate metrics
7. Display dashboard

---

## Default Scenarios

```text id="p6zopb"
spike
linear
noisy
periodic
```

---

## Custom Scenarios

```bash id="qpb88q"
python -m tests.run_experiments \
    --scenarios spike noisy \
    --seeds 1 2 3
```

---

## Save Results

```bash id="c7u1vv"
python -m tests.run_experiments --save
```

Creates:

```text id="zjlwm7"
results.json
```

---

## Custom Episode Length

```bash id="r76p54"
python -m tests.run_experiments \
    --steps 500
```

---

# 📉 Single-Episode Visualization

```bash id="cc5r4h"
python -m tests.run_single_episode
```

---

## What It Does

1. Load latest models
2. Create shared workload
3. Run:

```text id="ttmkam"
PPO
DQN
Baseline
```

4. Print metrics
5. Display time-series plot

---

## Custom Scenario

```bash id="pjxq7s"
python -m tests.run_single_episode \
    --scenario spike \
    --steps 500 \
    --seed 0
```

---

## Available Scenarios

```text id="yd2d8n"
default
spike
linear
noisy
periodic
```

---

## Another Example

```bash id="wh4m4u"
python -m tests.run_single_episode \
    --scenario periodic \
    --seed 42
```

---

# 🔬 Reward Diagnostic

```bash id="qym1tt"
python -m tests.diagnose_reward
```

---

## What It Does

Loads:

* PPO
* DQN

Runs:

```text id="g0bbkr"
Learned Policy
vs
Forced Scale-Up Policy
```

Prints:

* Mean reward
* Mean utilization
* Mean instance count
* Plain-language verdict

---

## Typical Output

```text id="z0cjlwm"
Policy Reward:        -0.42
Forced Reward:        -0.28
Verdict:
Training problem.
```

or

```text id="2nxdin"
Policy Reward:        -0.42
Forced Reward:        -0.55
Verdict:
Reward calibration problem.
```

---

# 🏗️ Design Philosophy

The project deliberately separates:

```text id="zjlwmv"
Evaluation Mechanics
```

from:

```text id="3ljrhr"
Experiment Design
```

This separation provides:

✅ Reusability

✅ Cleaner code

✅ Easier debugging

✅ Reproducible experiments

✅ Independent extension of evaluation and testing logic

---

# 🔑 Summary

The `tests/` module represents the experiment orchestration layer of the project. It is responsible for:

* Designing evaluation experiments
* Running multi-scenario comparisons
* Running multi-seed benchmarks
* Aggregating metrics
* Producing dashboards and visualizations
* Diagnosing reward-function issues
* Delivering the final performance results of PPO, DQN, and Baseline agents

Together with `evaluation/`, it forms the complete benchmarking framework used to validate RL-based cloud auto-scaling policies.