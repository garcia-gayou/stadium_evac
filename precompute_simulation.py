import os
import sys
import pickle
from simulation import Simulation
from multiprocessing import freeze_support

def run_simulation(name="simulation", num_agents=1000, max_frames=1000):
    sim = Simulation(num_agents=num_agents)
    output_dir = os.path.join("sim_outputs", name)
    os.makedirs(output_dir, exist_ok=True)

    frame = 0
    while not sim.is_finished() and frame < max_frames:
        sim.update()

        # Get both positions and pushover values
        data = sim.get_active_positions_and_pushovers()
        with open(os.path.join(output_dir, f"{frame:04d}.pkl"), "wb") as f:
            pickle.dump(data, f)

        frame += 1
        if frame % 5 == 0:
            print(f"Frame {frame} complete...")

    print(f"✅ Done! Total frames simulated: {frame}")

if __name__ == "__main__":
    freeze_support()

    name = sys.argv[1] if len(sys.argv) > 1 else "simulation"
    num_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    run_simulation(name=name, num_agents=num_agents)