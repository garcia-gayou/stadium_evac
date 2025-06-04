import numpy as np
from agents import Agent
from environment import Environment
from agent_distribution import generate_agent_positions
from scipy.spatial import KDTree

class Simulation:
    def __init__(self, num_agents=1000):
        self.env = Environment()
        self.agent_positions = generate_agent_positions(self.env, num_agents // 2, num_agents // 2)
        print(f"Requested: {num_agents} — Actually generated: {len(self.agent_positions)}")
        self.agents = [Agent(x, y) for (x, y) in self.agent_positions]
        self.finished = False

    def update(self):
        if self.finished:
            return

        # Mark agents as exited
        for agent in self.agents:
            if agent.has_exited:
                continue
            x, y = agent.position
            for (x0, y0), (x1, y1) in self.env.exits:
                if (x0 == x1 and abs(x - x0) < 0.15 and min(y0, y1) <= y <= max(y0, y1)) or \
                   (y0 == y1 and abs(y - y0) < 0.15 and min(x0, x1) <= x <= max(x0, x1)):
                    agent.has_exited = True
                    break

        # Get active agents
        active_agents = [a for a in self.agents if not a.has_exited]
        if len(active_agents) == 0:
            self.finished = True
            return

        # Prepare for KDTree neighbor lookup
        positions = np.array([a.position for a in active_agents])
        if positions.ndim != 2 or positions.shape[1] != 2:
            self.finished = True
            return

        tree = KDTree(positions)

        # Precompute neighbors for all agents
        neighbor_map = {}
        for i, agent in enumerate(active_agents):
            idxs = tree.query_ball_point(agent.position, r=3.0)
            neighbor_map[agent] = [active_agents[j] for j in idxs if active_agents[j] is not agent]

        # Step with neighbors
        for agent in active_agents:
            agent.step(neighbors=neighbor_map[agent], env=self.env)

        if all(agent.has_exited for agent in self.agents):
            self.finished = True

    def get_active_positions(self):
        return [agent.position for agent in self.agents if not agent.has_exited]

    def is_finished(self):
        return self.finished
