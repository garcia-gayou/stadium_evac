import numpy as np
import heapq

class Environment:
    def __init__(self, layout="none"):
        self.width = 130
        self.height = 100
        self.grid_res = 2.0  # finer resolution: 2 cells per meter
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
        self.obstacles = []
        self._generate_obstacles(layout)

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
            ((self.safe_left, self.divider_y), (self.safe_right, self.divider_y)),
            ((self.stage_left_x, self.stage_back_y), (self.stage_left_x, self.stage_front_y)),
            ((self.stage_left_x, self.stage_front_y), (self.stage_right_x, self.stage_front_y)),
            ((self.stage_right_x, self.stage_front_y), (self.stage_right_x, self.stage_back_y)),
        ]

    def _generate_obstacles(self, layout):
        if layout == "horizontal_barrier":
            # Strategic fence to split flow around exit (exit at x=59 to 71)
            fence_left = 59 + 1.5  # leave side gaps
            fence_right = 71 - 1.5
            y = 6.5  # ~6.5 meters above the exit
            self.obstacles.append(((fence_left, y), (fence_right, y)))
        elif layout == "left_block":
            self.obstacles.append(((10, 20), (10, 60)))
        elif layout == "cross_blocks":
            self.obstacles.append(((30, 40), (100, 40)))
            self.obstacles.append(((65, 20), (65, 70)))
        elif layout == "maze":
            for y in range(10, 90, 20):
                self.obstacles.append(((20, y), (110, y)))

    def _mark_cost_obstacles(self):
        for seg in self.walls + self.obstacles:
            (x0, y0), (x1, y1) = seg
            x0 = np.clip(int(round(x0 * self.grid_res)), 0, self.cost_grid.shape[0] - 1)
            x1 = np.clip(int(round(x1 * self.grid_res)), 0, self.cost_grid.shape[0] - 1)
            y0 = np.clip(int(round(y0 * self.grid_res)), 0, self.cost_grid.shape[1] - 1)
            y1 = np.clip(int(round(y1 * self.grid_res)), 0, self.cost_grid.shape[1] - 1)
            for x in range(min(x0, x1), max(x0, x1) + 1):
                for y in range(min(y0, y1), max(y0, y1) + 1):
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if abs(dy) == 1:
                                continue  # only expand horizontally
                            xx = x + dx
                            yy = y + dy
                            if 0 <= xx < self.cost_grid.shape[0] and 0 <= yy < self.cost_grid.shape[1]:
                                self.cost_grid[xx, yy] = np.inf

    def _compute_fast_marching_field(self):
        field = np.full_like(self.cost_grid, np.inf, dtype=float)
        visited = np.zeros_like(self.cost_grid, dtype=bool)
        frontier = []

        for x in range(self.cost_grid.shape[0]):
            for y in range(self.cost_grid.shape[1]):
                if self.cost_grid[x, y] == 0:
                    field[x, y] = 0
                    heapq.heappush(frontier, (0, x, y))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while frontier:
            cost, x, y = heapq.heappop(frontier)
            if visited[x, y]:
                continue
            visited[x, y] = True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cost_grid.shape[0] and 0 <= ny < self.cost_grid.shape[1]:
                    if not visited[nx, ny] and self.cost_grid[nx, ny] != np.inf:
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
        if norm > 0:
            return grad_sum / norm
        return np.array([0.0, 0.0])

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
        from numpy.linalg import norm
        p = np.array(point)
        a = np.array(line[0])
        b = np.array(line[1])
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        projection = a + t * ab
        return norm(p - projection)

    def get_obstacle_proximity(self, pos):
        """Return the shortest distance from pos to any wall or obstacle."""
        min_dist = float("inf")
        px, py = pos
        for (a, b) in self.walls + self.obstacles:
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