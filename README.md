# Agent-Based Evacuation Simulation — GNP Stadium

This project simulates the emergency evacuation of GNP Stadium in Mexico City using agent-based modeling and crowd dynamics based on the social force model. Each agent represents an individual with unique traits, navigating toward exits while avoiding collisions with other agents and obstacles.

The goal is to analyze crowd behavior under stress, identify bottlenecks, and evaluate evacuation protocols for large-scale events.

---

## Features

- Social force model (goal, repulsion, wall avoidance)
- Realistic agent traits: pushover, patience, frustration
- Crowd density bias in agent distribution
- Parallel simulation for high performance
- Animated visualization and metric plots
- Crush risk estimation using density and behavioral metrics

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point (optional extension base) |
| `simulation.py` | Core simulation loop |
| `agents.py` | Agent definitions and update logic |
| `environment.py` | Stadium geometry, walls, and exits |
| `agent_distribution.py` | Placement of agents with biased densities |
| `precompute_simulation.py` | Runs full simulation and logs output |
| `visualize_simulation.py` | Generates plots and animation |
| `sim_outputs/` | Folder where all simulation results are saved |

---

## Installation

Make sure you have **Python 3.8+** installed.

Install required libraries with:

```bash
pip install numpy matplotlib joblib scipy
```
## How to Run

### 1. Run a Simulation

Use the following command to simulate the evacuation of 1,000 agents:

```bash
python precompute_simulation.py run_1000 1000 (obstacle)
```

- `run_1000` is the name of the run (used as the folder name inside `sim_outputs/`)
- `1000` is the number of agents in the simulation
- The simulation will run until all agents have exited or the maximum frame limit is reached
- If you do not want an obstacle, you can run it without one, just skipping that part. The obstacle should be one of the following:
    - horizontal_barrier
    - funnel
    - parabola
- Output files (agent positions, crush index, and agent count) will be stored in:

`sim_outputs/run_1000/`

## 2. Visualize Results

To generate plots and an animated crowd movement video:

```bash
python visualize_simulation.py run_1000
```

The visual output includes:

- Remaining agents vs. time  
- Exit rate per second  
- Mean, max, and 95th percentile crowd densities  
- Crush risk index over time  
- Animation of the stadium with agent movement and pushover color mapping


