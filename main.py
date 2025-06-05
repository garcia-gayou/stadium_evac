from simulation import Simulation

if __name__ == "__main__":
    sim = Simulation(num_agents=1000)
    while not sim.is_finished():
        sim.update()