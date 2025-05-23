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

    fig, ax = plt.subplots()
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], color='blue')
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    def update(frame):
        for agent in agents:
            dx, dy = agent.position - env.exits[0]
            if not agent.has_exited and np.linalg.norm([dx, dy]) < exit_radius:
                agent.has_exited = True
        for agent in agents:
            agent.step(agents, env)

        active_positions = [a.position for a in agents if not a.has_exited]
        if active_positions:
            scat.set_offsets(np.array(active_positions).reshape(-1, 2))
        else:
            ani.event_source.stop()
        return scat,

    ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.show()
