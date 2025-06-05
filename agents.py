import math
import random

class Agent:
    def __init__(self, x, y, goal=None, radius=0.5, desired_speed=1.2, tau=0.3, pushover=None):
        self.position = [x, y]
        self.velocity = [0.0, 0.0]
        self.goal = goal if goal else [0.0, 0.0]
        self.radius = radius
        self.desired_speed = desired_speed
        self.tau = tau
        self.has_exited = False
        self.pushover = pushover if pushover is not None else min(max(random.gauss(0.5, 0.2), 0), 1)
        self.patience = max(random.gauss(self.pushover * 50, 5), 1)
        self.frustration = 0
        self.last_position = list(self.position)

        self.main_exit = None
        self.exit_options = []

    def choose_main_exit(self, env):
        min_dist = float('inf')
        chosen = None
        for exit_data in env.exits:
            exit_line = exit_data["points"]
            center = env.exit_centers[exit_line]
            if not self._can_reach(exit_line, env):
                continue
            dist = math.dist(self.position, center)
            if dist < min_dist:
                min_dist = dist
                chosen = exit_line
        if chosen:
            self.main_exit = chosen
            self.exit_options = env.exit_goals[chosen]

    def update_goal(self, env):
        if not self.exit_options:
            return
        best = min(self.exit_options, key=lambda g: math.dist(self.position, g))
        self.goal = best

    def compute_goal_force(self):
        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-5:
            return (0.0, 0.0)
        desired_vx = (dx / dist) * self.desired_speed
        desired_vy = (dy / dist) * self.desired_speed
        fx = (desired_vx - self.velocity[0]) / self.tau
        fy = (desired_vy - self.velocity[1]) / self.tau
        return fx * (1 + (1 - self.pushover)), fy * (1 + (1 - self.pushover))

    def compute_agent_repulsion(self, neighbors, A=10, B=0.8):
        fx, fy = 0.0, 0.0
        for other in neighbors:
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-5:
                continue
            n_x, n_y = dx / dist, dy / dist
            r_sum = self.radius + other.radius
            coeff = A * math.exp((r_sum - dist) / B)
            fx += self.pushover * 2.0 * coeff * n_x
            fy += self.pushover * 2.0 * coeff * n_y
        return fx, fy

    def compute_wall_repulsion(self, env, A=3, B=0.8, decay_scale=1.0):
        fx, fy = 0.0, 0.0
        for wall in env.walls:
            # Skip walls in exit zones
            skip = False
            for zone in env.exit_zones:
                if zone[0][0] <= self.position[0] <= zone[1][0] and \
                   zone[0][1] <= self.position[1] <= zone[1][1]:
                    skip = True
                    break
            if skip:
                continue

            # Normal repulsion from wall
            closest = [
                min(max(self.position[0], min(wall[0][0], wall[1][0])), max(wall[0][0], wall[1][0])),
                min(max(self.position[1], min(wall[0][1], wall[1][1])), max(wall[0][1], wall[1][1]))
            ]
            dx = self.position[0] - closest[0]
            dy = self.position[1] - closest[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-5:
                continue
            decay = math.exp(-dist / decay_scale)
            strength = A * math.exp((self.radius - dist) / B)
            fx += strength * (dx / dist) * (1 - decay)
            fy += strength * (dy / dist) * (1 - decay)
        return fx, fy

    def step_full(self, env, neighbors, dt=0.1):
        new_agent = self._clone()
        new_agent.update_goal(env)

        gx, gy = new_agent.compute_goal_force()
        ax, ay = new_agent.compute_agent_repulsion(neighbors)
        wx, wy = new_agent.compute_wall_repulsion(env)

        fx = gx + ax + wx
        fy = gy + ay + wy

        new_agent.velocity[0] += fx * dt
        new_agent.velocity[1] += fy * dt
        new_agent.position[0] += new_agent.velocity[0] * dt
        new_agent.position[1] += new_agent.velocity[1] * dt

        new_agent._check_patience()
        return new_agent

    def update_state_from(self, updated):
        self.position = updated.position
        self.velocity = updated.velocity
        self.last_position = updated.last_position
        self.frustration = updated.frustration
        self.goal = updated.goal

    def _clone(self):
        clone = Agent(*self.position)
        clone.velocity = list(self.velocity)
        clone.goal = list(self.goal)
        clone.radius = self.radius
        clone.desired_speed = self.desired_speed
        clone.tau = self.tau
        clone.pushover = self.pushover
        clone.patience = self.patience
        clone.frustration = self.frustration
        clone.last_position = list(self.last_position)
        clone.main_exit = self.main_exit
        clone.exit_options = list(self.exit_options)
        return clone

    def _check_patience(self, threshold=0.05):
        dx = self.position[0] - self.last_position[0]
        dy = self.position[1] - self.last_position[1]
        move = math.sqrt(dx * dx + dy * dy)
        if move < threshold:
            self.frustration += 1
        else:
            self.frustration = max(0, self.frustration - 0.5)
        self.last_position = list(self.position)

        if self.frustration > self.patience:
            dx = self.goal[0] - self.position[0]
            dy = self.goal[1] - self.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-5:
                self.velocity[0] += (dx / dist) * 0.8
                self.velocity[1] += (dy / dist) * 0.8

    def _can_reach(self, exit_line, env):
        y = self.position[1]
        ey0, ey1 = exit_line[0][1], exit_line[1][1]
        divider_y = env.divider_y
        return (y <= divider_y and ey0 <= divider_y and ey1 <= divider_y) or \
               (y > divider_y and ey0 > divider_y and ey1 > divider_y)
