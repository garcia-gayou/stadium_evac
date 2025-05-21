import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import Agent
from environment import Environment
import random

def run_simulation():
    env = Environment(width=20, height=20, exits=[(10, 0)])
    agents = [Agent(random.uniform(0, 20), random.uniform(10, 20), env.exits[0]) for _ in range(10)]

    fig, ax = plt.subplots()
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], color='blue')
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    def update(frame):
        for agent in agents:
            agent.move_toward_goal()
        scat.set_offsets([agent.position for agent in agents])
        return scat,

    ani = animation.FuncAnimation(fig, update, interval=100, blit=True)
    plt.show()
