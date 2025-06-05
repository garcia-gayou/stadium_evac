import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
import numpy as np

def visualize(name="simulation"):
    path = os.path.join("sim_outputs", name)
    if not os.path.exists(path):
        print(f"Folder '{path}' does not exist.")
        return

    files = sorted(f for f in os.listdir(path) if f.endswith(".pkl"))
    if not files:
        print(f"No simulation frames found in '{path}'.")
        return

    # Load frames of (position, pushover)
    frames = []
    for f in files:
        with open(os.path.join(path, f), "rb") as file:
            frames.append(pickle.load(file))  # list of (pos, pushover)

    # Setup environment for drawing walls/exits
    env = Environment()

    fig, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter([], [], s=10)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title(f"Evacuation Simulation: {name}")
    cmap = plt.get_cmap("coolwarm")

    # Draw walls
    for wall in env.walls:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=2)

    # Draw exits with thicker lines
    for exit_data in env.exits:
        (x0, y0), (x1, y1) = exit_data["points"]
        ax.plot([x0, x1], [y0, y1], color='green', linewidth=8, solid_capstyle='round')
        
    def update(i):
        data = frames[i]
        if not data:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))
            return scat,

        positions = [p for p, _ in data]
        pushovers = [p for _, p in data]
        norm_pushovers = [min(max(p, 0.0), 1.0) for p in pushovers]  # Ensure [0, 1] range

        scat.set_offsets(np.array(positions))
        scat.set_color([cmap(1 - p) for p in norm_pushovers])
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    visualize(name)