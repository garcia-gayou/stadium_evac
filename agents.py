import numpy as np

class Agent:
    def __init__(self, x, y, goal, radius=0.5, desired_speed=1.0, tau=0.3, pushover=None):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal, dtype=float)
        self.radius = radius
        self.desired_speed = desired_speed
        self.tau = tau
        self.has_exited = False
        self.pushover = pushover if pushover is not None else np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)

    def compute_goal_force(self):
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.zeros(2)
        desired_velocity = (direction / distance) * self.desired_speed
        base_force = (desired_velocity - self.velocity) / self.tau
        # More aggressive = stronger drive toward goal
        return base_force * (1 + (1 - self.pushover))

    def compute_agent_repulsion(self, agents, A=20, B=0.5):
        force = np.zeros(2)
        for other in agents:
            if other is self or other.has_exited:
                continue
            r_ij = self.radius + other.radius
            d_ij_vec = self.position - other.position
            d_ij = np.linalg.norm(d_ij_vec)
            if d_ij == 0:
                continue
            n_ij = d_ij_vec / d_ij
            repulsion = A * np.exp((r_ij - d_ij) / B) * n_ij
            # Pushovers feel stronger repulsion; jerks ignore it
            force += self.pushover * 2.0 * repulsion
        return force

    def compute_wall_repulsion(self, environment, A=5, B=0.5):
        force = np.zeros(2)
        for wall in environment.walls:
            closest_point = np.clip(self.position, wall[0], wall[1])
            d_iw_vec = self.position - closest_point
            d_iw = np.linalg.norm(d_iw_vec)
            if d_iw == 0:
                continue
            n_iw = d_iw_vec / d_iw
            repulsion = A * np.exp((self.radius - d_iw) / B) * n_iw
            force += repulsion
        return force

    def step(self, agents, environment, dt=0.1):
        if self.has_exited:
            return

        f_goal = self.compute_goal_force()
        f_agents = self.compute_agent_repulsion(agents)
        f_walls = self.compute_wall_repulsion(environment)

        total_force = f_goal + f_agents + f_walls
        acceleration = total_force  # assume mass = 1
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
