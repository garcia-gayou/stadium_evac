import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import Agent
from environment import Environment
import random
import numpy as np

exit_radius = 0.5
exit_rate_limit = 1  # max number of agents that can exit per frame
num_agents = 50
agent_speed = 0.5
agent_radius = 0.5

def run_simulation():
    env = Environment(width=20, height=20, exits=[(10, 0)])
    agents = [Agent(random.uniform(0, 20), random.uniform(10, 20), env.exits[0]) for _ in range(num_agents)]

    fig, ax = plt.subplots()
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], color='blue')
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    def update(frame):
        exiting_now = 0
        for agent in agents:
            if agent.has_exited:
                continue

            # Check if near the exit
            dx = agent.position[0] - env.exits[0][0]
            dy = agent.position[1] - env.exits[0][1]
            distance = (dx**2 + dy**2)**0.5

            if distance < exit_radius and exiting_now < exit_rate_limit:
                agent.has_exited = True
                exiting_now += 1
            else:
                agent.move_toward_goal(speed=0.2)

        active_positions = [a.position for a in agents if not a.has_exited]
        scat.set_offsets(np.array(active_positions).reshape(-1, 2))

        if all(agent.has_exited for agent in agents):
            ani.event_source.stop()
        return scat,

    ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.show()