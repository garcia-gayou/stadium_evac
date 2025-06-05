import numpy as np

class Environment:
    def __init__(self):
        self.width = 130
        self.height = 100
        self.divider_y = 50
        self.stage_width = 32
        self.stage_front_y = 90
        self.stage_back_y = self.height
        self.stage_left_x = (self.width - self.stage_width) / 2
        self.stage_right_x = (self.width + self.stage_width) / 2

        # Define exits with metadata
        self.exits = [
            {
                "points": ((self.width / 2 - 6.0, 0), (self.width / 2 + 6.0, 0)),
                "type": "bottom",
                "width": 12.0
            },
            {
                "points": ((0, 75), (0, 85)),
                "type": "side",
                "width": 2.5
            },
            {
                "points": ((self.width, 75), (self.width, 85)),
                "type": "side",
                "width": 2.5
            }
        ]

        self.walls = self._generate_walls()
        self.exit_goals = {}
        self.exit_centers = {}
        self.prepare_exit_goals()

        # New: bounding box zones around exits
        self.exit_zones = self._generate_exit_zones(buffer=1.0)

    def _generate_walls(self):
        exit_lines = [e["points"] for e in self.exits]
        (bx0, _), (bx1, _) = exit_lines[0]
        left_exit = exit_lines[1]
        right_exit = exit_lines[2]

        return [
            ((0, 0), (bx0, 0)),
            ((bx1, 0), (self.width, 0)),
            ((0, 0), (0, left_exit[0][1])),
            ((0, left_exit[1][1]), (0, self.height)),
            ((self.width, 0), (self.width, right_exit[0][1])),
            ((self.width, right_exit[1][1]), (self.width, self.height)),
            ((0, self.height), (self.width, self.height)),
            ((0, self.divider_y), (self.width, self.divider_y)),
            ((self.stage_left_x, self.stage_back_y), (self.stage_left_x, self.stage_front_y)),
            ((self.stage_left_x, self.stage_front_y), (self.stage_right_x, self.stage_front_y)),
            ((self.stage_right_x, self.stage_front_y), (self.stage_right_x, self.stage_back_y)),
        ]

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

            self.exit_goals[((x1, y1), (x2, y2))] = points
            self.exit_centers[((x1, y1), (x2, y2))] = np.mean([np.array([x1, y1]), np.array([x2, y2])], axis=0)

    def _generate_exit_zones(self, buffer=1.0):
        zones = []
        for exit_data in self.exits:
            (x1, y1), (x2, y2) = exit_data["points"]
            x_min = min(x1, x2) - buffer
            x_max = max(x1, x2) + buffer
            y_min = min(y1, y2) - buffer
            y_max = max(y1, y2) + buffer
            zones.append(((x_min, y_min), (x_max, y_max)))
        return zones
