import math
from environment import Environment
from agent_distribution import generate_agent_positions
from agents import Agent
from joblib import Parallel, delayed
from scipy.spatial import KDTree
import numpy as np

class Simulation:
    def __init__(self, num_agents=1000):
        self.env = Environment()
        self.env.prepare_exit_goals()
        self.agent_positions = generate_agent_positions(
            self.env, num_agents // 2, num_agents // 2
        )
        print(f"Requested: {num_agents} — Actually generated: {len(self.agent_positions)}")
        self.agents = [Agent(x, y) for x, y in self.agent_positions]
        for agent in self.agents:
            agent.choose_main_exit(self.env)
        self.finished = False
        self.frame = 0

    def update(self):
        if self.finished:
            return

        active_agents = [a for a in self.agents if not a.has_exited]
        if not active_agents:
            self.finished = True
            return

        positions = [a.position for a in active_agents]
        tree = KDTree(positions)

        # Neighbor map based on KDTree (within 3 meters)
        neighbor_map = {
            i: [active_agents[j] for j in tree.query_ball_point(pos, r=3.0) if i != j]
            for i, pos in enumerate(positions)
        }

        # Parallel execution of full agent step (including repulsion + updates)
        updated_agents = Parallel(n_jobs=-1)(
            delayed(agent.step_full)(self.env, neighbor_map[i])
            for i, agent in enumerate(active_agents)
        )

        # Commit updates to original agent objects
        for i, agent in enumerate(active_agents):
            agent.update_state_from(updated_agents[i])

        # Mark exited agents with improved 0.5m margin
        for agent in active_agents:
            x, y = agent.position
            for exit_data in self.env.exits:
                (x0, y0), (x1, y1) = exit_data["points"]
                # Exit is vertical
                if x0 == x1 and abs(x - x0) < 0.5 and min(y0, y1) <= y <= max(y0, y1):
                    agent.has_exited = True
                    break
                # Exit is horizontal
                elif y0 == y1 and abs(y - y0) < 0.5 and min(x0, x1) <= x <= max(x0, x1):
                    agent.has_exited = True
                    break

        self.frame += 1

    def get_active_positions_and_pushovers(self):
        return [(agent.position[:], agent.pushover) for agent in self.agents if not agent.has_exited]

    def is_finished(self):
        return self.finished
    
    def compute_crush_index(self):
        active = [a for a in self.agents if not a.has_exited]
        if not active:
            return 0.0

        positions = np.array([a.position for a in active])
        pushovers = np.array([a.pushover for a in active])
        
        # 1m x 1m grid bins
        H, _, _ = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=(self.env.width, self.env.height),
            range=[[0, self.env.width], [0, self.env.height]]
        )
        max_density = H.max()
        avg_pushover = pushovers.mean()
        return max_density * avg_pushover
