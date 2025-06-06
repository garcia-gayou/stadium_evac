import math
import random
import numpy as np
import copy

class Agent:
    def __init__(self, x, y, goal=None, radius=0.5, tau=0.3, pushover=None):
        self.position = [x, y]
        self.velocity = [0.0, 0.0]
        self.goal = goal if goal else [0.0, 0.0]
        self.radius = radius
        self.tau = tau
        self.has_exited = False

        self.pushover = pushover if pushover is not None else np.clip(random.gauss(0.5, 0.2), 0, 1)
        self.initial_pushover = self.pushover
        self.patience = max(random.gauss(self.pushover * 50, 5), 1)
        self.frustration = 0
        self.last_position = list(self.position)

        base_speed = random.gauss(1.3, 0.25)
        self.base_speed = base_speed
        self.desired_speed = min(max(base_speed + 0.2 * (self.pushover - 0.5), 1.0), 2.75)

        self.main_exit = None
        self.exit_options = []
        self.exit_targets = []

        self._last_proximity = None
        self._last_proximity_position = list(self.position)

        self.stuck_counter = 0
        self._last_checked_position = list(self.position)


    def choose_main_exit(self, env):
        accessible = env.get_accessible_exits(self.position)
        if not accessible:
            raise ValueError(f"No reachable exits for agent at {self.position}")
        def dist_to_line(line): return env.distance_to_line(self.position, line)
        best_line = min(accessible, key=dist_to_line)
        self.main_exit = best_line
        self.exit_options = accessible
        self.exit_targets = env.exit_goals[best_line]
        self.goal = random.choice(self.exit_targets)

    def update_goal(self, env):
        if self.exit_targets:
            self.goal = random.choice(self.exit_targets)
        else:
            grad = self.compute_fmm_gradient(env)
            self.goal = [self.position[0] + grad[0], self.position[1] + grad[1]]

    def compute_fmm_gradient(self, env):
        if env.obstacle_points is None or len(env.obstacle_points) == 0:
            return np.array([0.0, 0.0])
        min_dist = np.min(np.linalg.norm(env.obstacle_points - self.position, axis=1))
        if min_dist < 10.0:
            return env.get_fmm_gradient(self.position[0], self.position[1])
        return np.array([0.0, 0.0])

    def get_cached_obstacle_proximity(self, env, threshold=0.1):
        dx = self.position[0] - self._last_proximity_position[0]
        dy = self.position[1] - self._last_proximity_position[1]
        moved = math.sqrt(dx * dx + dy * dy)
        if self._last_proximity is not None and moved < threshold:
            return self._last_proximity
        self._last_proximity = env.get_obstacle_proximity(self.position)
        self._last_proximity_position = list(self.position)
        return self._last_proximity

    def compute_goal_force(self, env):
        gradient_strength = env.get_gradient_magnitude(self.position[0], self.position[1])

        proximity = self.get_cached_obstacle_proximity(env)
        blend_weight = min(0.85, 1 / (1 + math.exp(1.5 * (proximity - 4.0))))

        if gradient_strength < 0.05:
            blend_weight *= 0.25  # trust less the global field

        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-5:
            return (0.0, 0.0)

        desired_vx = (dx / dist) * self.desired_speed
        desired_vy = (dy / dist) * self.desired_speed

        flow = self.compute_fmm_gradient(env)
        blended = np.array([
            (1 - blend_weight) * desired_vx + blend_weight * flow[0] * self.desired_speed,
            (1 - blend_weight) * desired_vy + blend_weight * flow[1] * self.desired_speed
        ])

        fx = (blended[0] - self.velocity[0]) / self.tau
        fy = (blended[1] - self.velocity[1]) / self.tau

        # If stuck for many frames, inject jitter
        if self.stuck_counter >= 10:
            jitter = np.random.uniform(-1, 1, size=2)
            jitter /= np.linalg.norm(jitter)
            jitter *= 0.5  # Small push
            fx += jitter[0]
            fy += jitter[1]

        # Detour assist: if close to other agents or walls, bias lateral movement
        proximity = self.get_cached_obstacle_proximity(env)
        if proximity < 1.5:  # near wall or obstacle
            side_step = np.array([-dy, dx])  # perpendicular vector
            side_step = side_step / np.linalg.norm(side_step)
            fx += 0.2 * side_step[0]
            fy += 0.2 * side_step[1]

        # Barrier-side sliding: add lateral bias when stuck and near wall
        if self.stuck_counter >= 8:
            proximity = self.get_cached_obstacle_proximity(env)
            if proximity < 1.5:
                # Compute perpendicular direction to goal vector (for sideways move)
                dx = self.goal[0] - self.position[0]
                dy = self.goal[1] - self.position[1]
                norm = math.sqrt(dx * dx + dy * dy)
                if norm > 1e-3:
                    lateral = np.array([-dy, dx]) / norm
                    direction = random.choice([-1, 1])  # Left or right slide
                    fx += 0.2 * direction * lateral[0]
                    fy += 0.2 * direction * lateral[1]

        if gradient_strength < 0.01 and self.stuck_counter >= 5:
            # Direct movement toward goal (ignore FMM)
            dx = self.goal[0] - self.position[0]
            dy = self.goal[1] - self.position[1]
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 1e-5:
                fx = (dx / norm * self.desired_speed - self.velocity[0]) / self.tau
                fy = (dy / norm * self.desired_speed - self.velocity[1]) / self.tau

        return fx, fy

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
        # Slight asymmetry to break deadlocks
        bias = np.random.uniform(-0.05, 0.05)
        fx += bias

        return fx, fy

    def compute_wall_repulsion(self, env, A=10, B=0.7):
        fx, fy = 0.0, 0.0
        for wall in env.walls:
            a, b = wall
            p = self.position
            abx, aby = b[0] - a[0], b[1] - a[1]
            apx, apy = p[0] - a[0], p[1] - a[1]
            ab_len_sq = abx * abx + aby * aby
            if ab_len_sq == 0:
                closest = a
            else:
                t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
                closest = [a[0] + t * abx, a[1] + t * aby]
            dx = p[0] - closest[0]
            dy = p[1] - closest[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-5:
                continue
            nx, ny = dx / dist, dy / dist
            tx, ty = -ny, nx
            strength = A * math.exp((self.radius - dist) / B)
            tangential = (self.velocity[0] * tx + self.velocity[1] * ty)
            fx += strength * nx - 0.3 * tangential * tx
            fy += strength * ny - 0.3 * tangential * ty
        return fx, fy

    def compute_obstacle_repulsion(self, env, A=10, B=0.7):
        fx, fy = 0.0, 0.0
        for obs in env.obstacles:
            a, b = obs
            p = self.position
            abx, aby = b[0] - a[0], b[1] - a[1]
            apx, apy = p[0] - a[0], p[1] - a[1]
            ab_len_sq = abx * abx + aby * aby
            if ab_len_sq == 0:
                closest = a
            else:
                t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
                closest = [a[0] + t * abx, a[1] + t * aby]
            dx = p[0] - closest[0]
            dy = p[1] - closest[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-5:
                continue
            nx, ny = dx / dist, dy / dist
            tx, ty = -ny, nx
            strength = A * math.exp((self.radius - dist) / B)
            tangential = (self.velocity[0] * tx + self.velocity[1] * ty)
            fx += strength * nx - 0.5 * tangential * tx
            fy += strength * ny - 0.5 * tangential * ty

        return fx, fy

    def step_full(self, env, neighbors, dt=0.1):
        new_agent = self._clone()
        new_agent.update_goal(env)
        gx, gy = new_agent.compute_goal_force(env)
        ax, ay = new_agent.compute_agent_repulsion(neighbors)
        wx, wy = new_agent.compute_wall_repulsion(env)
        ox, oy = new_agent.compute_obstacle_repulsion(env)
        fx = gx + ax + wx + ox
        fy = gy + ay + wy + oy
        new_agent.velocity[0] += fx * dt
        new_agent.velocity[1] += fy * dt
        speed = math.sqrt(new_agent.velocity[0]**2 + new_agent.velocity[1]**2)
        if speed > 3.0:
            scale = 3.0 / speed
            new_agent.velocity[0] *= scale
            new_agent.velocity[1] *= scale
        new_agent.position[0] += new_agent.velocity[0] * dt
        new_agent.position[1] += new_agent.velocity[1] * dt
        new_agent._check_patience()
        return new_agent

    def _check_patience(self, threshold=0.05):
        dx = self.position[0] - self.last_position[0]
        dy = self.position[1] - self.last_position[1]
        move = math.sqrt(dx * dx + dy * dy)
        self.frustration += 1 if move < threshold else -0.5
        self.frustration = max(0, self.frustration)

        frustration_time = self.frustration * 0.1
        boost = min(1.0, max(0.0, (frustration_time - 5) / 10.0))
        self.desired_speed = min(self.base_speed * (1 + 0.35 * boost), 2.75)

        t_start = 5 + 0.1 * self.patience
        t_end = t_start + 10 + 0.2 * self.patience
        duration = t_end - t_start

        if frustration_time >= t_start:
            blend = (1 - math.cos(math.pi * min(frustration_time - t_start, duration) / duration)) / 2
            target = min(self.initial_pushover, 0.1)
            self.pushover = (1 - blend) * self.initial_pushover + blend * target

        self.last_position = list(self.position)

    def update_state_from(self, updated):
        self.position = updated.position
        # Track if the agent has barely moved
        dx = self.position[0] - self._last_checked_position[0]
        dy = self.position[1] - self._last_checked_position[1]
        movement = math.sqrt(dx * dx + dy * dy)

        if movement < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self._last_checked_position = list(self.position)

        self.velocity = updated.velocity
        self.last_position = updated.last_position
        self.frustration = updated.frustration
        self.goal = updated.goal
        self.pushover = updated.pushover
        self.desired_speed = updated.desired_speed

    def _clone(self):
        return copy.deepcopy(self) 

