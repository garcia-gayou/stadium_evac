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

    remaining_counts_path = os.path.join(path, "remaining_counts.pkl")

    # Setup environment for drawing walls/exits
    env = Environment()

    if os.path.exists(remaining_counts_path):
        with open(remaining_counts_path, "rb") as f:
            remaining_counts = pickle.load(f)

        # Plot graph
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
    else:
        print("⚠️ No remaining_counts.pkl found. Skipping time-vs-people-left plot.")

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
    else:
        print("⚠️ No exit_rate.pkl found. Skipping exit rate plot.")

    # --- Density over time ---
    density_over_time = []
    mean_densities = []
    perc95_densities = []


    for frame_data in frames:
        if not frame_data:
            density_over_time.append(0)
            continue

        positions = np.array([p for p, _ in frame_data])

        # Use 1m x 1m bins
        H, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=(env.width, env.height),  # 1m resolution if your environment is in meters
            range=[[0, env.width], [0, env.height]]
        )

        flat = H.flatten()
        max_density = flat.max()
        mean_density = flat.mean()
        perc95_density = np.percentile(flat, 95)

        density_over_time.append(max_density)
        mean_densities.append(mean_density)
        perc95_densities.append(perc95_density)


    dt = 0.1
    time_axis = np.arange(len(mean_densities)) * dt

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