import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import Agent
from environment import Environment
import random
import numpy as np

num_agents = 50

def run_simulation(num_agents=20, exit_radius=1.0):
    env = Environment(width=20, height=20, exits=[(10, 0)])
    agents = [Agent(random.uniform(0, 20), random.uniform(10, 20), env.exits[0]) for _ in range(num_agents)]

    # Create scatter plot with color mapping
    pushover_values = [a.pushover for a in agents]
    cmap = plt.get_cmap('coolwarm')  # blue (nice) to red (jerk)
    colors = [cmap(1 - p) for p in pushover_values]  # invert so 0=red, 1=blue

    fig, ax = plt.subplots()
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], c=colors)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title("Social Force Simulation (color = pushover)")

    def update(frame):
        for agent in agents:
            if not agent.has_exited and np.linalg.norm(agent.position - env.exits[0]) < exit_radius:
                agent.has_exited = True

        for agent in agents:
            agent.step(agents, env)

        active_positions = [a.position for a in agents if not a.has_exited]
        active_colors = [colors[i] for i, a in enumerate(agents) if not a.has_exited]

        if active_positions:
            scat.set_offsets(np.array(active_positions).reshape(-1, 2))
            scat.set_color(active_colors)
        else:
            ani.event_source.stop()
        return scat,

    ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.show()