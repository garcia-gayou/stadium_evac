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

    files = sorted(f for f in os.listdir(path) if f.endswith(".pkl") and f[:4].isdigit())
    if not files:
        print(f"No simulation frames found in '{path}'.")
        return

    # Load frames of (position, pushover)
    frames = []
    for f in files:
        with open(os.path.join(path, f), "rb") as file:
            frames.append(pickle.load(file))  # list of (pos, pushover)

    # Load layout name if available
    layout_path = os.path.join(path, "layout.txt")
    layout = name
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            layout = f.read().strip()

    env = Environment(layout=layout)
    plt.figure(figsize=(10, 8))
    plt.imshow(env.fmm_field.T, origin='lower', cmap='plasma')
    plt.colorbar(label='FMM Cost')
    plt.title("FMM Cost Field (plasma colormap)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow((env.cost_grid == np.inf).T, cmap="gray_r", origin="lower")
    plt.title("Obstacle and Wall Mask (white = blocked)")
    plt.tight_layout()
    plt.show()

    # Plot static graphs
    remaining_counts_path = os.path.join(path, "remaining_counts.pkl")
    if os.path.exists(remaining_counts_path):
        with open(remaining_counts_path, "rb") as f:
            remaining_counts = pickle.load(f)
        dt = 0.1
        time_axis = np.arange(len(remaining_counts)) * dt
        plt.figure()
        plt.plot(time_axis, remaining_counts, label="People Left", color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("Number of People Remaining")
        plt.title("Evacuation Progress Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    exit_rate_path = os.path.join(path, "exit_rate.pkl")
    if os.path.exists(exit_rate_path):
        with open(exit_rate_path, "rb") as f:
            exit_rate = pickle.load(f)
        time_axis = np.arange(len(exit_rate)) * 0.1
        plt.figure()
        plt.plot(time_axis, exit_rate, label="People Exiting per Second", color='green')
        plt.xlabel("Time (s)")
        plt.ylabel("Exit Rate (agents/s)")
        plt.title("Exit Throughput Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Density
    density_over_time = []
    mean_densities = []
    perc95_densities = []
    for frame_data in frames:
        if not frame_data:
            density_over_time.append(0)
            continue
        positions = np.array([p for p, _ in frame_data])
        H, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=(env.width, env.height),
            range=[[0, env.width], [0, env.height]]
        )
        flat = H.flatten()
        density_over_time.append(flat.max())
        mean_densities.append(flat.mean())
        perc95_densities.append(np.percentile(flat, 95))

    time_axis = np.arange(len(mean_densities)) * 0.1
    plt.figure()
    plt.plot(time_axis, mean_densities, label="Mean Density", color='blue')
    plt.plot(time_axis, perc95_densities, label="95th Percentile Density", color='orange')
    plt.plot(time_axis, density_over_time[:len(time_axis)], label="Max Local Density", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Density (people/m²)")
    plt.title("Crowd Density Statistics Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Crush index
    crush_index_path = os.path.join(path, "crush_index.pkl")
    if os.path.exists(crush_index_path):
        with open(crush_index_path, "rb") as f:
            crush_indices = pickle.load(f)
        time_axis = np.arange(len(crush_indices)) * 0.1
        plt.figure()
        plt.plot(time_axis, crush_indices, label="Crush Index", color='purple')
        plt.xlabel("Time (s)")
        plt.ylabel("Crush Index")
        plt.title("Crush Risk Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Static scene setup
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title(f"Evacuation Simulation: {name}")

    # Draw walls
    for wall in env.walls:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=2)

    # Draw exits
    for exit_data in env.exits:
        (x0, y0), (x1, y1) = exit_data["points"]
        ax.plot([x0, x1], [y0, y1], color='green', linewidth=8, solid_capstyle='round')

    # Draw obstacles
    for (x1, y1), (x2, y2) in env.obstacles:
        ax.plot([x1, x2], [y1, y2], color='orange', linewidth=2)

    # ▶️ FMM field visualization (quiver arrows)
    step = 1  # Arrow spacing
    X, Y, U, V = [], [], [], []
    for x in range(0, int(env.width), step):
        for y in range(0, int(env.height), step):
            grad = env.get_fmm_gradient(x, y)
            if np.linalg.norm(grad) > 0:
                X.append(x)
                Y.append(y)
                U.append(grad[0])
                V.append(grad[1])

    # Much smaller and lighter arrows
    ax.quiver(X, Y, U, V, color='grey', alpha=0.4, scale=60, width=0.001)
    # Agents (animated)
    scat = ax.scatter([], [], s=10)
    cmap = plt.get_cmap("coolwarm")

    def update(i):
        data = frames[i]
        if not data:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))
            return scat,

        positions = [p for p, _ in data]
        pushovers = [p for _, p in data]
        norm_pushovers = [min(max(p, 0.0), 1.0) for p in pushovers]
        scat.set_offsets(np.array(positions))
        scat.set_color([cmap(1 - p) for p in norm_pushovers])
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    visualize(name)
