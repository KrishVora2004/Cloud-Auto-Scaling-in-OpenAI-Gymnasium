# 🚀 RL-Based Cloud Auto-Scaling

A reinforcement learning system for intelligent and cost-efficient cloud auto-scaling. The project trains **PPO** and **DQN** agents to dynamically adjust the number of active server instances in response to changing workloads, aiming to minimize infrastructure costs while maintaining acceptable service-level performance.

---

## 📖 Table of Contents

* [Overview](#-overview)
* [Architecture](#-architecture)
* [Project Structure](#-project-structure)
* [How It Works](#-how-it-works)
* [Design Rationale](#-design-rationale)
* [Advantages of RL-Based Auto-Scaling](#-advantages-of-rl-based-auto-scaling)
* [Technology Stack](#-technology-stack)
* [Quick Start](#-quick-start)
* [Evaluation Scenarios](#-evaluation-scenarios)
* [Key Results](#-key-results)

---

## 📖 Overview

Cloud environments experience highly dynamic and unpredictable workloads. Choosing the correct number of active instances is a continuous balancing act:

* **Under-provisioning** leads to SLA violations, increased latency, and dropped requests.
* **Over-provisioning** wastes infrastructure resources and increases operational costs.

Traditional threshold-based auto-scalers react to current utilization but cannot systematically optimize multiple competing objectives.

This project models cloud auto-scaling as a **Markov Decision Process (MDP)** and uses **Reinforcement Learning (RL)** to learn adaptive scaling policies that outperform conventional rule-based approaches.

---

## 🏗️ Architecture

### High-Level Flow

```text
Workload Generator
        │
        ▼
 Cloud Simulator
        │
        ▼
 Gymnasium Environment
        │
        ▼
 PPO / DQN Agents
        │
        ▼
 Evaluation & Testing
```

---

## 📂 Project Structure

```text
project/
│
├── sim/          Cloud simulator and workload generator
├── envs/         Gymnasium environment wrapper
├── agents/       PPO and DQN training pipelines
├── evaluation/   Baseline policy and evaluation utilities
├── tests/        Experiment runner and dashboard generation
├── models/       Saved trained model checkpoints
└── results/      Generated evaluation outputs
```

### Dependency Flow

```text
sim/ ← envs/ ← agents/
                  │
                  ▼
             evaluation/ ← tests/
```

**Module Responsibilities**

| Module        | Responsibility                                      |
| ------------- | --------------------------------------------------- |
| `sim/`        | Cloud dynamics, queueing model, workload generation |
| `envs/`       | RL-compatible Gymnasium interface                   |
| `agents/`     | PPO and DQN training                                |
| `evaluation/` | Baseline comparison and model evaluation            |
| `tests/`      | Multi-scenario experiments and dashboards           |
| `models/`     | Saved trained policies                              |

---

## ⚙️ How It Works

At each discrete timestep:

1. The workload generator produces the incoming request rate `λₜ`.
2. The RL agent observes the current cloud state.
3. The agent selects one of three actions:

   * **0** → Scale Down
   * **1** → Maintain Current Capacity
   * **2** → Scale Up
4. The simulator updates the number of active instances `Nₜ`.
5. Performance metrics are computed using an M/M/1 queueing model:

   * CPU Utilization
   * Response Time
   * Error Rate
   * Infrastructure Cost
6. A reward is generated based on these metrics.
7. The agent updates its policy to maximize long-term reward.

---

## 🧠 Design Rationale

### Markov Decision Process (MDP)

Auto-scaling naturally fits an MDP framework because it contains:

* Observable system state
* Discrete control actions
* Stochastic workload transitions
* A measurable optimization objective

This formulation enables the use of modern reinforcement learning algorithms without requiring custom optimization techniques.

### Simulator–Environment Separation

The simulator remains completely independent of RL.

```text
CloudSimulator
      ↓
Gym Environment
      ↓
RL Agent
```

Benefits:

* Easier testing
* Cleaner architecture
* Future extensibility
* Independent simulator improvements

### Mixed-Scenario Training

Agents are trained across multiple workload patterns:

* Default
* Spike
* Linear Growth
* Noisy
* Periodic

This improves policy robustness and prevents overfitting to a single traffic pattern.

### Idle-Capacity Penalty

The reward function explicitly penalizes unnecessary idle resources, encouraging:

✅ Cost-efficient scaling

✅ Aggressive scale-down during low demand

✅ Sufficient capacity during workload spikes

---

## ✅ Advantages of RL-Based Auto-Scaling

### 💰 Lower Operational Cost

RL agents learn when resources are genuinely required, reducing unnecessary instance usage.

### 📈 Better Generalization

A single trained policy can handle multiple workload types without manual retuning.

### ⚖️ Multi-Objective Optimization

The reward function simultaneously considers:

* Cost
* Latency
* SLA Violations
* Resource Efficiency

### 🔮 Proactive Scaling

Unlike threshold policies that only react after utilization changes, RL agents can learn workload patterns and scale ahead of demand.

### 📊 Reproducible Evaluation

All approaches are evaluated under identical conditions:

* Same workload traces
* Same metrics
* Same evaluation procedures

This ensures fair comparison between PPO, DQN, and baseline policies.

---

## 🛠️ Technology Stack

| Component              | Technology        |
| ---------------------- | ----------------- |
| Reinforcement Learning | Stable-Baselines3 |
| Algorithms             | PPO, DQN          |
| Environment Interface  | Gymnasium         |
| Numerical Computing    | NumPy             |
| Queueing Model         | M/M/1             |
| Visualization          | Matplotlib        |
| Language               | Python 3.11+      |

---

## 🚀 Quick Start

### 1️⃣ Train Agents

Train both PPO and DQN:

```bash
python -m agents.train --algo both --steps 200000
```

Train a specific algorithm:

```bash
python -m agents.train --algo ppo --steps 200000
python -m agents.train --algo dqn --steps 200000
```

---

### 2️⃣ Visualize a Single Episode

```bash
python -m tests.run_single_episode
```

---

### 3️⃣ Run Evaluation

Generate metrics and comparison dashboard:

```bash
python -m tests.run_experiments
```

---

### 4️⃣ Evaluate Specific Scenarios (Optional)

```bash
python -m tests.run_single_episode --scenario spike --steps 500 --seed 0
```

```bash
python -m tests.run_experiments \
    --scenarios spike noisy \
    --seeds 1 2 3 \
    --save
```

---

## 🌊 Evaluation Scenarios

The system is tested against several workload patterns:

| Scenario | Description                 |
| -------- | --------------------------- |
| Default  | Mixed realistic workload    |
| Spike    | Sudden traffic bursts       |
| Linear   | Gradually increasing demand |
| Noisy    | Random fluctuations         |
| Periodic | Repeating cyclic patterns   |

These scenarios help evaluate both adaptability and robustness.

---


## 🎯 Project Goal

Develop an intelligent cloud resource management system capable of:

* Minimizing infrastructure cost
* Maintaining SLA compliance
* Reducing response latency
* Adapting to diverse workload patterns
* Demonstrating the effectiveness of reinforcement learning for cloud resource optimization

---