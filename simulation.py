# Updated simulation.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import Agent
from environment import Environment
import random
import numpy as np

def run_simulation(num_agents=20):
    env = Environment()
    agents = [Agent(random.uniform(5, 35), random.uniform(5, 35), np.array([20, 0])) for _ in range(num_agents)]

    pushover_values = [a.pushover for a in agents]
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(1 - p) for p in pushover_values]

    fig, ax = plt.subplots()
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], c=colors)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title("Social Force Simulation (Stadium Layout)")

    # Draw walls
    for wall in env.walls:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=2)

    # Draw exits
    for exit in env.exits:
        ax.plot([exit[0][0], exit[1][0]], [exit[0][1], exit[1][1]], color='green', linewidth=3)

    finished = [False]
    delay_counter = [0]

    def update(frame):
        for agent in agents:
            if not agent.has_exited:
                x, y = agent.position
                for (x0, y0), (x1, y1) in env.exits:
                    if (x0 == x1 and abs(x - x0) < 0.5 and min(y0, y1) <= y <= max(y0, y1)) or \
                       (y0 == y1 and abs(y - y0) < 0.5 and min(x0, x1) <= x <= max(x0, x1)):
                        agent.has_exited = True
                        break

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
