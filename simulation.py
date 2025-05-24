import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import Agent
from environment import Environment
import random
import numpy as np

def run_simulation(num_agents=20, exit_radius=1.0):
    env = Environment(width=20, height=20, exits=[(10, 0)])
    agents = [Agent(random.uniform(0, 20), random.uniform(10, 20), env.exits[0]) for _ in range(num_agents)]

    # Color coding by pushover
    pushover_values = [a.pushover for a in agents]
    cmap = plt.get_cmap('coolwarm')  # blue = nice, red = jerk
    colors = [cmap(1 - p) for p in pushover_values]

    fig, ax = plt.subplots()
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], c=colors)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title("Social Force Simulation (color = pushover)")

    finished = [False]
    delay_counter = [0]

    def update(frame):
        for agent in agents:
            if not agent.has_exited and np.linalg.norm(agent.position - env.exits[0]) < exit_radius:
                agent.has_exited = True

        for agent in agents:
            agent.step(agents, env)

        active_positions = [a.position for a in agents if not a.has_exited]
        active_colors = [colors[i] for i, a in enumerate(agents) if not a.has_exited]

        scat.set_offsets(np.array(active_positions).reshape(-1, 2))
        scat.set_color(active_colors)

        if all(agent.has_exited for agent in agents):
            if not finished[0]:
                finished[0] = True
                delay_counter[0] = 10
            elif delay_counter[0] > 0:
                delay_counter[0] -= 1
            else:
                ani.event_source.stop()

        return scat,

    ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.show()
