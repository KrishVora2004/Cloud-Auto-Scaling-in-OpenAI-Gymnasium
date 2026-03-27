# Environments

This module defines the custom simulation environment used for training reinforcement learning agents for cloud resource optimization.

It models how system metrics (like workload, utilization, response time, and errors) evolve over time and how agent actions (e.g., scaling instances) affect system performance.

---

## Overview

The `envs` module provides:

- **Custom Gym Environment**  
  A fully defined RL environment compatible with OpenAI Gym/Gymnasium standards.

- **State Representation**  
  Encodes the system state using key metrics such as:
  - Workload (incoming requests)
  - Instance count
  - CPU/Resource utilization
  - Response time
  - Error rate

- **Action Space**  
  Defines possible actions the agent can take, typically:
  - Scale up resources
  - Scale down resources
  - Maintain current state

- **Reward Function**  
  Guides learning by balancing:
  - Performance (low latency, low error rate)
  - Cost (number of active instances)
  - Stability (avoiding unnecessary scaling)

---

## Key Components

- **Environment Class**  
  Implements core Gym methods:
  - `reset()` → Initializes the environment state  
  - `step(action)` → Applies action and returns next state, reward, done flag  
  - `render()` (optional) → Visualization/debugging  

- **Dynamics Simulation**  
  Models how workload impacts:
  - Resource utilization
  - Response time
  - System failures

- **Constraints Handling**  
  Ensures:
  - Instance count remains within valid bounds  
  - No invalid system states occur  

---

## Purpose

This environment acts as the testing ground where RL agents learn optimal scaling strategies under varying workloads.

---

## Notes

- Designed to be deterministic or stochastic depending on workload generation.
- Can be extended to include more realistic cloud metrics.
- Compatible with PPO and DQN agents without modification.
