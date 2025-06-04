import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import Agent
from environment import Environment
from agent_distribution import generate_agent_positions
import numpy as np

def run_simulation(num_agents=20000):
    env = Environment()
    agent_positions = generate_agent_positions(env, num_agents // 2, num_agents // 2)
    print(f"🎯 Requested: {num_agents} — Actually generated: {len(agent_positions)}")
    agents = [Agent(x, y) for (x, y) in agent_positions]

    pushover_values = [a.pushover for a in agents]
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(1 - p) for p in pushover_values]

    fig, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter([a.position[0] for a in agents],
                      [a.position[1] for a in agents], c=colors, s=10)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title("Social Force Simulation (Stadium Layout in Meters)")

    for wall in env.walls:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=1.5)

    for exit in env.exits:
        ax.plot([exit[0][0], exit[1][0]], [exit[0][1], exit[1][1]], color='green', linewidth=3)

    finished = [False]
    delay_counter = [0]

    def update(frame):
        for agent in agents:
            if not agent.has_exited:
                x, y = agent.position
                for (x0, y0), (x1, y1) in env.exits:
                    if (x0 == x1 and abs(x - x0) < 0.15 and min(y0, y1) <= y <= max(y0, y1)) or \
                       (y0 == y1 and abs(y - y0) < 0.15 and min(x0, x1) <= x <= max(x0, x1)):
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

    plt.show()
    return  # Skip animation for now
    ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()