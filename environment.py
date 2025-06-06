import numpy as np
import heapq

class Environment:
    def __init__(self, layout="none", curve_resolution=10):
        self.width = 130
        self.height = 100
        self.grid_res = 6.0
        self.grid_width = int(self.width * self.grid_res)
        self.grid_height = int(self.height * self.grid_res)

        self.divider_y = 50
        self.stage_width = 32
        self.stage_front_y = 90
        self.stage_back_y = self.height
        self.stage_left_x = (self.width - self.stage_width) / 2
        self.stage_right_x = (self.width + self.stage_width) / 2

        self.safe_left = 0.0 + 1e-6
        self.safe_right = self.width - 1e-6

        self.exits = [
            {"points": ((self.width / 2 - 6.0, 0), (self.width / 2 + 6.0, 0)), "type": "bottom", "width": 12.0},
            {"points": ((0, 75), (0, 85)), "type": "side", "width": 2.5},
            {"points": ((self.width, 75), (self.width, 85)), "type": "side", "width": 2.5}
        ]

        self.walls = self._generate_walls()
        self.divider = ((self.safe_left, self.divider_y), (self.safe_right, self.divider_y))
        self.walls.append(self.divider)  # Include in agent repulsion only

        self.obstacles = []
        self._generate_obstacles(layout, curve_resolution)
        self.obstacle_points = self._get_all_obstacle_points()

        self.cost_grid = np.ones((self.grid_width, self.grid_height))
        self.exit_goals = {}
        self.exit_centers = {}
        self.prepare_exit_goals()

        self._mark_cost_obstacles()
        self.fmm_field = self._compute_fast_marching_field()

    def _generate_walls(self):
        exit_lines = [e["points"] for e in self.exits]
        (bx0, _), (bx1, _) = exit_lines[0]
        left_exit = exit_lines[1]
        right_exit = exit_lines[2]

        return [
            ((self.safe_left, 0), (bx0, 0)),
            ((bx1, 0), (self.safe_right, 0)),
            ((self.safe_left, 0), (self.safe_left, left_exit[0][1])),
            ((self.safe_left, left_exit[1][1]), (self.safe_left, self.height)),
            ((self.safe_right, 0), (self.safe_right, right_exit[0][1])),
            ((self.safe_right, right_exit[1][1]), (self.safe_right, self.height)),
            ((self.safe_left, self.height), (self.safe_right, self.height)),
            ((self.stage_left_x, self.stage_back_y), (self.stage_left_x, self.stage_front_y)),
            ((self.stage_left_x, self.stage_front_y), (self.stage_right_x, self.stage_front_y)),
            ((self.stage_right_x, self.stage_front_y), (self.stage_right_x, self.stage_back_y)),
        ]

    def _generate_obstacles(self, layout, curve_resolution=10):
        if layout == "horizontal_barrier":
            self.obstacles.append(((60, 7.5), (70, 7.5)))
        elif layout == "left_block":
            self.obstacles.append(((10, 20), (10, 60)))
        elif layout == "cross_blocks":
            self.obstacles.append(((30, 40), (100, 40)))
            self.obstacles.append(((65, 20), (65, 70)))
        elif layout == "maze":
            for y in range(10, 90, 20):
                self.obstacles.append(((20, y), (110, y)))
        elif layout == "parabola":
            width = 30
            clearance = 10
            peak_height = 20
            x1 = (self.width - width) / 2
            x2 = (self.width + width) / 2
            f = fit_parabola(x1, x2, clearance, clearance, peak_height)
            x_vals = np.linspace(x1, x2, curve_resolution)
            points = [(x, f(x)) for x in x_vals]
            for i in range(len(points) - 1):
                self.obstacles.append((points[i], points[i + 1]))
        elif layout == "funnel":
            exit_left = 59
            exit_right = 71
            exit_y = 0
            gap_from_exit = 6.0
            funnel_top_y = 12.0
            funnel_apex_x = (exit_left + exit_right) / 2
            base_left = (exit_left, exit_y + gap_from_exit)
            base_right = (exit_right, exit_y + gap_from_exit)
            apex = (funnel_apex_x, funnel_top_y)
            self.obstacles.append((apex, base_left))
            self.obstacles.append((apex, base_right))

    def _get_all_obstacle_points(self):
        points = []
        for obs in self.obstacles:
            (x0, y0), (x1, y1) = obs
            steps = max(int(np.linalg.norm([x1 - x0, y1 - y0]) * 10), 1)
            for i in range(steps + 1):
                t = i / steps
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                points.append([x, y])
        return np.array(points)

    def _mark_cost_obstacles(self):
        high_cost = 100  # effectively solid but allows FMM gradient computation

        # Add both obstacles and the divider
        for seg in self.obstacles + [self.divider]:
            (x0, y0), (x1, y1) = seg
            x0 = np.clip(int(round(x0 * self.grid_res)), 0, self.cost_grid.shape[0] - 1)
            x1 = np.clip(int(round(x1 * self.grid_res)), 0, self.cost_grid.shape[0] - 1)
            y0 = np.clip(int(round(y0 * self.grid_res)), 0, self.cost_grid.shape[1] - 1)
            y1 = np.clip(int(round(y1 * self.grid_res)), 0, self.cost_grid.shape[1] - 1)

            for x in range(min(x0, x1), max(x0, x1) + 1):
                for y in range(min(y0, y1), max(y0, y1) + 1):
                    # Expand to 3x3 neighborhood to cover adjacent cells
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            xx, yy = x + dx, y + dy
                            if 0 <= xx < self.cost_grid.shape[0] and 0 <= yy < self.cost_grid.shape[1]:
                                self.cost_grid[xx, yy] = high_cost

    def _compute_fast_marching_field(self):
        field = np.full_like(self.cost_grid, np.inf, dtype=float)
        visited = np.zeros_like(self.cost_grid, dtype=bool)
        frontier = []

        for x in range(self.cost_grid.shape[0]):
            for y in range(self.cost_grid.shape[1]):
                if self.cost_grid[x, y] == 0:
                    field[x, y] = 0
                    heapq.heappush(frontier, (0, x, y))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while frontier:
            cost, x, y = heapq.heappop(frontier)
            if visited[x, y]:
                continue
            visited[x, y] = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cost_grid.shape[0] and 0 <= ny < self.cost_grid.shape[1]:
                    if not visited[nx, ny] and np.isfinite(self.cost_grid[nx, ny]):
                        new_cost = cost + self.cost_grid[nx, ny]
                        if new_cost < field[nx, ny]:
                            field[nx, ny] = new_cost
                            heapq.heappush(frontier, (new_cost, nx, ny))
        return field

    def get_fmm_gradient(self, x, y):
        xi = np.clip(int(x * self.grid_res), 0, self.fmm_field.shape[0] - 1)
        yi = np.clip(int(y * self.grid_res), 0, self.fmm_field.shape[1] - 1)
        grad_candidates = []
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (-1, 1), (1, -1)
        ]
        current = self.fmm_field[xi, yi]
        if not np.isfinite(current):
            return np.array([0.0, 0.0])
        for dx, dy in directions:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < self.fmm_field.shape[0] and 0 <= ny < self.fmm_field.shape[1]:
                neighbor = self.fmm_field[nx, ny]
                if np.isfinite(neighbor):
                    vec = np.array([dx, dy])
                    diff = neighbor - current
                    grad = -diff * vec
                    grad_candidates.append(grad)
        if not grad_candidates:
            return np.array([0.0, 0.0])
        grad_sum = np.sum(grad_candidates, axis=0)
        norm = np.linalg.norm(grad_sum)
        return grad_sum / norm if norm > 0 else np.array([0.0, 0.0])

    def prepare_exit_goals(self, num_points=5, margin_ratio=0.1):
        for exit_data in self.exits:
            (x1, y1), (x2, y2) = exit_data["points"]
            length = np.linalg.norm([x2 - x1, y2 - y1])
            margin = margin_ratio * length
            vec = np.array([x2 - x1, y2 - y1])
            unit_vec = vec / np.linalg.norm(vec)
            start = np.array([x1, y1]) + unit_vec * margin
            end = np.array([x2, y2]) - unit_vec * margin
            points = [start + t * (end - start) for t in np.linspace(0, 1, num_points)]

            key = ((x1, y1), (x2, y2))
            self.exit_goals[key] = points
            self.exit_centers[key] = np.mean([np.array([x1, y1]), np.array([x2, y2])], axis=0)

            for point in points:
                xi = int(np.floor(point[0] * self.grid_res))
                yi = int(np.floor(point[1] * self.grid_res))
                xi = np.clip(xi, 0, self.cost_grid.shape[0] - 1)
                yi = np.clip(yi, 0, self.cost_grid.shape[1] - 1)
                self.cost_grid[xi, yi] = 0

    def get_accessible_exits(self, pos):
        above_divider = pos[1] >= self.divider_y
        accessible = []
        for e in self.exits:
            (p1, p2) = e["points"]
            center_y = (p1[1] + p2[1]) / 2
            if (above_divider and center_y >= self.divider_y) or (not above_divider and center_y < self.divider_y):
                accessible.append((p1, p2))
        return accessible

    def distance_to_line(self, point, line):
        p = np.array(point)
        a, b = np.array(line[0]), np.array(line[1])
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    def get_obstacle_proximity(self, pos):
        min_dist = float("inf")
        px, py = pos
        for (a, b) in self.obstacles:
            ax, ay = a
            bx, by = b
            ab = np.array([bx - ax, by - ay])
            ap = np.array([px - ax, py - ay])
            ab_norm = np.dot(ab, ab)
            if ab_norm == 0:
                closest = np.array(a)
            else:
                t = np.clip(np.dot(ap, ab) / ab_norm, 0, 1)
                closest = np.array(a) + t * ab
            dist = np.linalg.norm(np.array(pos) - closest)
            if dist < min_dist:
                min_dist = dist
        return min_dist


def fit_parabola(x1, x2, y1, y2, y_mid):
    x_mid = (x1 + x2) / 2
    A = np.array([
        [x1**2, x1, 1],
        [x_mid**2, x_mid, 1],
        [x2**2, x2, 1]
    ])
    Y = np.array([y1, y_mid, y2])
    a, b, c = np.linalg.solve(A, Y)
    return lambda x: a * x**2 + b * x + c
