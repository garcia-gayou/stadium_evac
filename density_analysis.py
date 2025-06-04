import numpy as np
import matplotlib.pyplot as plt

from agents import Agent
from environment import Environment
from agent_distribution import generate_agent_positions

def run_density_analysis(num_agents=20000, sample_squares=1000, square_size=1.0):
    env = Environment()
    agent_positions = generate_agent_positions(env, num_agents // 2, num_agents // 2)
    print(f"🎯 Requested: {num_agents} — Actually generated: {len(agent_positions)}")

    agents = [Agent(x, y) for (x, y) in agent_positions]

    # Generate 100 random square meter sample regions
    np.random.seed(42)
    sample_centers = np.column_stack((
        np.random.uniform(0, env.width - square_size, sample_squares),
        np.random.uniform(0, env.height - square_size, sample_squares)
    ))

    # Count agents in each square
    densities = []
    for (cx, cy) in sample_centers:
        count = sum(
            (cx <= a.position[0] < cx + square_size) and
            (cy <= a.position[1] < cy + square_size)
            for a in agents
        )
        densities.append(count)

    # Plot histogram of densities
    plt.hist(densities, bins=range(0, max(densities) + 2), edgecolor='black', align='left')
    plt.axvline(x=4, color='orange', linestyle='--', label='4 people/m² (dense)')
    plt.axvline(x=5, color='red', linestyle='--', label='5 people/m² (very dense)')
    plt.xlabel('Agents per 1m² square')
    plt.ylabel('Frequency')
    plt.title('Local Crowd Density Sampled from 1000 Random Squares')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

run_density_analysis()