import numpy as np

class Environment:
    def __init__(self):
        self.width = 130
        self.height = 100
        self.exit_width = 3
        self.side_exit_height = 10
        self.divider_y = 50
        self.stage_width = 32
        self.stage_front_y = 90
        self.stage_back_y = self.height
        self.stage_left_x = (self.width - self.stage_width) / 2
        self.stage_right_x = (self.width + self.stage_width) / 2

        self.exits = [
            ((self.width / 2 - self.exit_width / 2, 0), (self.width / 2 + self.exit_width / 2, 0)),
            ((0, 75), (0, 85)),
            ((self.width, 75), (self.width, 85)),
        ]

        self.walls = [
            ((0, 0), (self.width / 2 - self.exit_width / 2, 0)),
            ((self.width / 2 + self.exit_width / 2, 0), (self.width, 0)),
            ((0, 0), (0, 75)),
            ((0, 85), (0, self.height)),
            ((self.width, 0), (self.width, 75)),
            ((self.width, 85), (self.width, self.height)),
            ((0, self.height), (self.width, self.height)),
            ((0, self.divider_y), (self.width, self.divider_y)),
            ((self.stage_left_x, self.stage_back_y), (self.stage_left_x, self.stage_front_y)),
            ((self.stage_left_x, self.stage_front_y), (self.stage_right_x, self.stage_front_y)),
            ((self.stage_right_x, self.stage_front_y), (self.stage_right_x, self.stage_back_y)),
        ]

        self.exit_goals = {}
        self.exit_centers = {}

    def prepare_exit_goals(self, num_points=5, margin_ratio=0.1):
        for (x1, y1), (x2, y2) in self.exits:
            length = np.linalg.norm([x2 - x1, y2 - y1])
            margin = margin_ratio * length
            vec = np.array([x2 - x1, y2 - y1])
            unit_vec = vec / np.linalg.norm(vec)
            start = np.array([x1, y1]) + unit_vec * margin
            end = np.array([x2, y2]) - unit_vec * margin
            points = [start + t * (end - start) for t in np.linspace(0, 1, num_points)]
            self.exit_goals[((x1, y1), (x2, y2))] = points

            # Save center for choosing nearest exit
            self.exit_centers[((x1, y1), (x2, y2))] = ((x1 + x2) / 2, (y1 + y2) / 2)
