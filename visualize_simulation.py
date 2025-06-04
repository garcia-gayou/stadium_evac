import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment import Environment
import numpy as np

# Load simulation data
run_id = sys.argv[1] if len(sys.argv) > 1 else "default"
data_path = f"precomputed_simulation/positions_{run_id}.pkl"

with open(data_path, "rb") as f:
    positions_per_frame = pickle.load(f)

# Set up plot and environment
env = Environment()
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=5)

ax.set_xlim(0, env.width)
ax.set_ylim(0, env.height)
ax.set_title(f"Simulation Playback — {run_id}")

# Draw walls
for wall in env.walls:
    (x1, y1), (x2, y2) = wall
    ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.5)

# Draw exits
for ex in env.exits:
    (x1, y1), (x2, y2) = ex
    ax.plot([x1, x2], [y1, y2], color="green", linewidth=3)

# Animation
def update(frame):
    positions = positions_per_frame[frame]
    if positions:
        x, y = zip(*positions)
        scat.set_offsets(list(zip(x, y)))
    else:
        scat.set_offsets(np.empty((0, 2)))
    return scat,

ani = FuncAnimation(fig, update, frames=len(positions_per_frame), interval=50, blit=True)
plt.show()
