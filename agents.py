import math
import random
import numpy as np
import copy

class Agent:
    def __init__(self, x, y, goal=None, radius=0.3, tau=0.3, pushover=None):
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

        self.frustration_scale = random.uniform(0.07, 0.12)
        self.pushover_min_target = random.uniform(0.05, 0.15)

        self.in_wiggle_mode = False
        self.wiggle_mode_timer = 0.0
        self.wiggle_mode_duration = 0.0

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

    def compute_fmm_gradient(self, env):
        if env.obstacle_points is None or len(env.obstacle_points) == 0:
            return np.array([0.0, 0.0])
        min_dist = np.min(np.linalg.norm(env.obstacle_points - self.position, axis=1))
        if min_dist < 10.0:
            return env.get_fmm_gradient(self.position[0], self.position[1])
        return np.array([0.0, 0.0])

    def update_goal(self, env):
        if self.exit_targets:
            def dist_to(p): return np.linalg.norm(np.array(self.position) - np.array(p))
            self.goal = min(self.exit_targets, key=dist_to)
        else:
            grad = self.compute_fmm_gradient(env)
            self.goal = [self.position[0] + grad[0], self.position[1] + grad[1]]

    def get_cached_obstacle_proximity(self, env, threshold=0.1):
        dx = self.position[0] - self._last_proximity_position[0]
        dy = self.position[1] - self._last_proximity_position[1]
        moved = math.sqrt(dx * dx + dy * dy)
        if self._last_proximity is not None and moved < threshold:
            return self._last_proximity
        self._last_proximity = env.get_obstacle_proximity(self.position)
        self._last_proximity_position = list(self.position)
        return self._last_proximity

    def compute_goal_force(self, env, dt):
        proximity = self.get_cached_obstacle_proximity(env)
        blend_weight = min(0.5, 1 / (1 + math.exp(2.0 * (proximity - 3.0))))  # 0% @ 5m, 75% @ 1m

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

        # Enter wiggle mode if frustrated and not already in it
        if self.frustration > 50 and not self.in_wiggle_mode:
            self.in_wiggle_mode = True
            self.wiggle_mode_timer = 0.0
            self.wiggle_mode_duration = min(2.0 + 0.02 * self.frustration, 5.0)

        # If in wiggle mode, override normal goal force
        if self.in_wiggle_mode:
            self.wiggle_mode_timer += dt
            if self.wiggle_mode_timer >= self.wiggle_mode_duration:
                self.in_wiggle_mode = False
            else:
                return self.compute_frustrated_wiggle_force(env)

        return fx, fy
    
    def compute_frustrated_wiggle_force(self, env):
        # Default fallback direction if no good obstacle is found
        default_direction = np.array([0.0, -1.0])

        nearest_dist = float("inf")
        best_segment = None

        for obs in env.obstacles:
            a, b = np.array(obs[0]), np.array(obs[1])
            ab = b - a
            ap = np.array(self.position) - a
            ab_len_sq = np.dot(ab, ab)

            if ab_len_sq == 0:
                continue  # degenerate segment

            t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
            closest = a + t * ab
            dist = np.linalg.norm(np.array(self.position) - closest)

            if dist < nearest_dist:
                nearest_dist = dist
                best_segment = (a, b)

        # If no close obstacle found, fallback to default
        if best_segment is None or nearest_dist > 2.0:
            direction = default_direction
        else:
            a, b = best_segment
            tangent = b - a
            tangent = tangent / np.linalg.norm(tangent)

            # Try both directions of the tangent
            step = 0.1
            pos_plus = np.array(self.position) + tangent * step
            pos_minus = np.array(self.position) - tangent * step

            # Choose the one that moves down in Y
            direction = tangent if pos_plus[1] < pos_minus[1] else -tangent

        # Scale by frustration-based strength
        max_force = 20.0
        offset = 120.0
        steepness = 30.0
        raw = 1 / (1 + math.exp(-(self.frustration - offset) / steepness))
        strength = max_force * raw

        fx, fy = direction * strength
        return fx, fy

    def compute_agent_repulsion(self, neighbors, A=7.5, B=0.5, max_range=2.0):
        fx, fy = 0.0, 0.0
        for other in neighbors:
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-5 or dist > max_range:
                continue
            n_x, n_y = dx / dist, dy / dist
            r_sum = self.radius + other.radius
            # Logistic-based repulsion strength
            strength = A / (1 + math.exp((dist - r_sum) / B))
            fx += self.pushover * 2 * strength * n_x
            fy += self.pushover * 2 * strength * n_y
        return fx, fy

    def compute_wall_repulsion(self, env, A=10.0, B=0.4, max_range=1.5):
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

            dx, dy = p[0] - closest[0], p[1] - closest[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-5 or dist > max_range:
                continue

            nx, ny = dx / dist, dy / dist
            strength = A / (1 + math.exp((dist - self.radius) / B))
            fx += strength * nx 
            fy += strength * ny
        return fx, fy

    def compute_obstacle_repulsion(self, env, A=17.5, B=0.5, max_range=1.5):
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
            dist = math.hypot(dx, dy)
            if dist < 1e-5 or dist > max_range:
                continue

            nx, ny = dx / dist, dy / dist
            strength = A / (1 + math.exp((dist - self.radius) / B))
            fx += strength * nx
            fy += strength * ny
        return fx, fy

    def step_full(self, env, neighbors, dt=0.1):
        new_agent = self._clone()
        new_agent.update_goal(env)
        gx, gy = new_agent.compute_goal_force(env, dt)
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
        expected = self.desired_speed * 0.1 * 0.4

        # Frustration increase if agent is stuck
        if move < expected:
            self.frustration += random.uniform(0.5, 1.0)
        else:
            # Random de-escalation rate
            deescalation = random.uniform(1.0, 2.0)
            self.frustration -= deescalation

        self.frustration = max(0, self.frustration)

        frustration_time = self.frustration * self.frustration_scale
        boost = min(1.0, max(0.0, (frustration_time - 4) / 8.0))
        self.desired_speed = self.base_speed * (1 + 0.35 * boost)

        t_start = 4 + 0.1 * self.patience
        t_end = t_start + 8 + 0.15 * self.patience
        duration = t_end - t_start

        if frustration_time >= t_start:
            blend = (1 - math.cos(math.pi * min(frustration_time - t_start, duration) / duration)) / 2
            target = self.pushover_min_target
            self.pushover = (1 - blend) * self.initial_pushover + blend * target

        self.last_position = list(self.position)

    def update_state_from(self, updated):
        self.position = updated.position
        self.velocity = updated.velocity
        self.last_position = updated.last_position
        self.frustration = updated.frustration
        self.goal = updated.goal
        self.pushover = updated.pushover
        self.desired_speed = updated.desired_speed

    def _clone(self):
        return copy.deepcopy(self)