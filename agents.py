import numpy as np

class Agent:
    def __init__(self, x, y, goal=None, radius=0.5, desired_speed=1.0, tau=0.3, pushover=None):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal) if goal is not None else np.zeros(2)
        self.radius = radius
        self.desired_speed = desired_speed
        self.tau = tau
        self.has_exited = False

        self.pushover = pushover if pushover is not None else np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
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

        if distance < 1.0:
            return base_force * 2.0

        aggressiveness_boost = 1 + (1 - self.pushover)
        return base_force * aggressiveness_boost

    def compute_agent_repulsion(self, agents, A=20, B=0.5):
        force = np.zeros(2)
        for other in agents:
            if other is self or other.has_exited:
                continue
            r_ij = self.radius + other.radius
            d_ij_vec = self.position - other.position
            d_ij = np.linalg.norm(d_ij_vec)
            if d_ij < 1e-5:
                continue
            n_ij = d_ij_vec / d_ij
            repulsion = A * np.exp((r_ij - d_ij) / B) * n_ij
            force += self.pushover * 2.0 * repulsion
        return force

    def compute_wall_repulsion(self, environment, A=5, B=0.5, decay_scale=1.0):
        force = np.zeros(2)
        for wall in environment.walls:
            closest_point = np.clip(self.position, wall[0], wall[1])
            d_iw_vec = self.position - closest_point
            d_iw = np.linalg.norm(d_iw_vec)
            if d_iw < 1e-5:
                continue
            n_iw = d_iw_vec / d_iw

            min_exit_dist = min(
                self.distance_to_line_segment(closest_point, e0, e1)
                for (e0, e1) in environment.exits
            ) if environment.exits else float('inf')

            decay = np.exp(-min_exit_dist / decay_scale)
            strength = A * np.exp((self.radius - d_iw) / B)
            repulsion = strength * n_iw * (1 - decay)
            force += repulsion
        return force

    @staticmethod
    def distance_to_line_segment(p, a, b):
        a = np.array(a)
        b = np.array(b)
        ap = p - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def check_patience(self, threshold=0.05):
        displacement = np.linalg.norm(self.position - self.last_position)
        if displacement < threshold:
            self.frustration += 1
        else:
            self.frustration = max(self.frustration - 0.5, 0)
        self.last_position = np.copy(self.position)

        if self.frustration > self.patience:
            direction = self.goal - self.position
            distance = np.linalg.norm(direction)
            if distance > 1e-5:
                direction /= distance
                self.velocity += direction * 0.8

    def step(self, agents, environment, dt=0.1):
        if self.has_exited:
            return

        exit_goals = environment.get_exit_goals(num_points=7)

        # Agents should not choose exits that are on the other side of the middle wall
        if self.position[1] > 20:
            # Agent is on upper side — exclude bottom exit
            filtered_goals = [g for g in exit_goals if g[1] >= 20]
        else:
            # Agent is on lower side — exclude upper exits
            filtered_goals = [g for g in exit_goals if g[1] <= 20]

        # If no filtered goals remain, fallback to all (just in case)
        filtered_goals = filtered_goals if filtered_goals else exit_goals

        def goal_score(g):
            dist = np.linalg.norm(self.position - g)
            wall_dist = min(
                self.distance_to_line_segment(g, w0, w1)
                for (w0, w1) in environment.walls
            )
            penalty = 0.5 if wall_dist < 0.5 else 0.0  # steer away from walls
            return dist + penalty

        self.goal = min(filtered_goals, key=goal_score)

        f_goal = self.compute_goal_force()
        f_agents = self.compute_agent_repulsion(agents)
        f_walls = self.compute_wall_repulsion(environment)

        total_force = f_goal + f_agents + f_walls
        acceleration = total_force
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Prevent escape through non-exit walls
        x, y = self.position
        if y < 0:
            allowed = any(
                y0 == 0 and y1 == 0 and x0 <= x <= x1
                for (x0, y0), (x1, y1) in environment.exits
            )
            if not allowed:
                self.position[1] = 0

        self.position[0] = np.clip(self.position[0], 0, environment.width)
        self.position[1] = np.clip(self.position[1], 0, environment.height)

        self.check_patience()
