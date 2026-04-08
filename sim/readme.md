# Simulation

This module is responsible for generating and managing the workload dynamics used within the environment. It simulates how user demand changes over time, which directly impacts system behavior and agent decisions.

The simulation acts as the driving force behind the environment by producing realistic and varied workload patterns.

---

## Overview

The `sim` module provides:

- **Workload Generation**  
  Simulates incoming request patterns over time.

- **Dynamic Behavior**  
  Models fluctuations such as:
  - Sudden spikes (traffic bursts)
  - Gradual increases/decreases
  - Periodic patterns

- **Time-Step Progression**  
  Ensures that workload evolves step-by-step during each environment interaction.

---

## Key Components

- **Workload Generator**  
  Produces workload values (`λ_t`) at each timestep, which serve as input to the environment.

- **Pattern Modeling**  
  Supports different workload types:
  - Constant load  
  - Random variations  
  - Sinusoidal / periodic load  
  - Spike-based scenarios  

- **Noise Handling**  
  Adds randomness to better simulate real-world unpredictability.

---

## Purpose

The simulation module provides realistic input conditions for the RL agents to learn from. By exposing agents to varying workload scenarios, it helps them develop robust and adaptive scaling strategies.

---

## Notes

- Can be extended to include real-world traffic traces.
- Plays a critical role in testing generalization of trained agents.
- Designed to be lightweight and easily configurable.