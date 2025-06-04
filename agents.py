import numpy as np
from scipy.spatial import KDTree

class Agent:
    def __init__(self, x, y, goal=None, radius=0.5, desired_speed=1.0, tau=0.3, pushover=None):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal) if goal is not None else np.zeros(2)
        self.radius = radius
        self.desired_speed = desired_speed
        self.tau = tau
        self.has_exited = False

        self.pushover = pushover if pushover is not None else np.clip(np.random.normal(0.5, 0.2), 0, 1)
        self.patience = np.random.normal(loc=self.pushover * 50, scale=5)
        self.frustration = 0
        self.last_position = np.copy(self.position)

    def compute_goal_force(self):
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance < 1e-5:
            return np.zeros(2)
        if distance < 0.3:
            return (direction / distance) * self.desired_speed * 0.5
        desired_velocity = (direction / distance) * self.desired_speed
        base_force = (desired_velocity - self.velocity) / self.tau
        return base_force * (1 + (1 - self.pushover))

    def compute_agent_repulsion(self, neighbors, A=20, B=0.5):
        force = np.zeros(2)
        for other in neighbors:
            d_vec = self.position - other.position
            d = np.linalg.norm(d_vec)
            if d < 1e-5:
                continue
            n = d_vec / d
            r_sum = self.radius + other.radius
            repulsion = A * np.exp((r_sum - d) / B) * n
            force += self.pushover * 2.0 * repulsion
        return force

    def compute_wall_repulsion(self, env, A=5, B=0.5, decay_scale=1.0):
        force = np.zeros(2)
        for wall in env.walls:
            closest = np.clip(self.position, wall[0], wall[1])
            d_vec = self.position - closest
            d = np.linalg.norm(d_vec)
            if d < 1e-5:
                continue
            n = d_vec / d
            min_exit_dist = min(
                self.distance_to_line_segment(closest, e0, e1) for (e0, e1) in env.exits
            ) if env.exits else float('inf')
            decay = np.exp(-min_exit_dist / decay_scale)
            strength = A * np.exp((self.radius - d) / B)
            force += strength * n * (1 - decay)
        return force

    @staticmethod
    def distance_to_line_segment(p, a, b):
        a, b = np.array(a), np.array(b)
        ap = p - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def check_patience(self, threshold=0.05):
        move = np.linalg.norm(self.position - self.last_position)
        self.frustration += 1 if move < threshold else -0.5
        self.frustration = max(0, self.frustration)
        self.last_position = np.copy(self.position)
        if self.frustration > self.patience:
            dir = self.goal - self.position
            d = np.linalg.norm(dir)
            if d > 1e-5:
                self.velocity += (dir / d) * 0.8

    def step(self, neighbors, env, dt=0.1):
        if self.has_exited:
            return

        exit_goals = env.get_exit_goals(num_points=7)
        divider_y = env.divider_y if hasattr(env, 'divider_y') else env.height / 2

        if self.position[1] > divider_y:
            exit_options = [g for g in exit_goals if g[1] > divider_y]
        else:
            exit_options = [g for g in exit_goals if g[1] < divider_y]
        if not exit_options:
            exit_options = exit_goals

        def goal_score(g):
            dist = np.linalg.norm(self.position - g)
            wall_dist = min(
                self.distance_to_line_segment(g, w0, w1) for (w0, w1) in env.walls
            )
            return dist + (0.5 if wall_dist < 0.5 else 0)

        self.goal = min(exit_options, key=goal_score)

        f_goal = self.compute_goal_force()
        f_agents = self.compute_agent_repulsion(neighbors)
        f_walls = self.compute_wall_repulsion(env)

        total_force = f_goal + f_agents + f_walls
        self.velocity += total_force * dt
        self.position += self.velocity * dt

        self.position[0] = np.clip(self.position[0], 0, env.width)
        self.position[1] = np.clip(self.position[1], 0, env.height)

        self.check_patience()
