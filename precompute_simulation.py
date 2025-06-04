import pickle
import os
import sys
from simulation import Simulation

# Accept a run name from the command line (optional)
run_id = sys.argv[1] if len(sys.argv) > 1 else "default"

# Ensure the output directory exists
output_dir = "precomputed_simulation"
os.makedirs(output_dir, exist_ok=True)

# Set file path
output_path = os.path.join(output_dir, f"positions_{run_id}.pkl")

# Run and store simulation frames
MAX_FRAMES = 1000
sim = Simulation(num_agents=1000)
positions_per_frame = []

for i in range(MAX_FRAMES):
    sim.update()
    positions = [tuple(agent.position) for agent in sim.agents if not agent.has_exited]
    positions_per_frame.append(positions)

    if i % 5 == 0:
        print(f"Saved frame {i}/{MAX_FRAMES}")

# Save the data
with open(output_path, "wb") as f:
    pickle.dump(positions_per_frame, f)

print(f"✅ Simulation saved to {output_path}")