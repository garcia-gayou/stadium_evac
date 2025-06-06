import os
import sys
import pickle
from simulation import Simulation
from multiprocessing import freeze_support

def run_simulation(name="simulation", num_agents=1000, layout="none", max_frames=2000):
    sim = Simulation(num_agents=num_agents, layout=layout)    
    output_dir = os.path.join("sim_outputs", name)
    os.makedirs(output_dir, exist_ok=True)

    frame = 0
    remaining_counts = []
    crush_indices = []

    while not sim.is_finished() and frame < max_frames:
        sim.update()
        data = sim.get_active_positions_and_pushovers()
        remaining_counts.append(len(data))
        
        # Store crush index
        crush_indices.append(sim.compute_crush_index())
        
        with open(os.path.join(output_dir, f"{frame:04d}.pkl"), "wb") as f:
            pickle.dump(data, f)
        
        frame += 1
        if frame % 5 == 0:
            print(f"Frame {frame} complete...")

    with open(os.path.join(output_dir, "crush_index.pkl"), "wb") as f:
        pickle.dump(crush_indices, f)

    with open(os.path.join(output_dir, "remaining_counts.pkl"), "wb") as f:
        pickle.dump(remaining_counts, f)
    # Compute people exiting per frame
    exit_rate = []
    for i in range(1, len(remaining_counts)):
        exited = remaining_counts[i - 1] - remaining_counts[i]
        exit_rate.append(exited)

    # Save exit rate
    with open(os.path.join(output_dir, "exit_rate.pkl"), "wb") as f:
        pickle.dump(exit_rate, f)

    with open(os.path.join(output_dir, "layout.txt"), "w") as f:
        f.write(layout)

    print(f"✅ Done! Total frames simulated: {frame}")
    return frame

    

if __name__ == "__main__":
    freeze_support()

    name = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    num_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    layout = sys.argv[3] if len(sys.argv) > 3 else "none"

    run_simulation(name=name, num_agents=num_agents, layout=layout)

