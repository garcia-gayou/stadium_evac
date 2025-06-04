import numpy as np

class Environment:
    def __init__(self):
        # Stadium dimensions in meters
        self.width = 130     # meters
        self.height = 100    # meters

        # Key layout dimensions (all in meters)
        self.exit_width = 3
        self.side_exit_height = 10
        self.divider_y = 50   # Midway horizontal divider (e.g., fence)
        self.stage_width = 32
        self.stage_front_y = 90  # Stage front (audience side)
        self.stage_back_y = self.height
        self.stage_left_x = (self.width - self.stage_width) / 2
        self.stage_right_x = (self.width + self.stage_width) / 2

        # Exits
        self.exits = [
            # Bottom center
            ((self.width / 2 - self.exit_width / 2, 0),
             (self.width / 2 + self.exit_width / 2, 0)),
            # Left middle
            ((0, 75), (0, 85)),
            # Right middle
            ((self.width, 75), (self.width, 85)),
        ]

        # Walls (outer boundary, divider, stage structure)
        self.walls = [
            # Bottom wall (excluding center exit)
            ((0, 0), (self.width / 2 - self.exit_width / 2, 0)),
            ((self.width / 2 + self.exit_width / 2, 0), (self.width, 0)),

            # Left wall (excluding exit gap)
            ((0, 0), (0, 75)),
            ((0, 85), (0, self.height)),

            # Right wall (excluding exit gap)
            ((self.width, 0), (self.width, 75)),
            ((self.width, 85), (self.width, self.height)),

            # Top wall
            ((0, self.height), (self.width, self.height)),

            # Divider fence
            ((0, self.divider_y), (self.width, self.divider_y)),

            # Stage (U-shape)
            ((self.stage_left_x, self.stage_back_y), (self.stage_left_x, self.stage_front_y)),
            ((self.stage_left_x, self.stage_front_y), (self.stage_right_x, self.stage_front_y)),
            ((self.stage_right_x, self.stage_front_y), (self.stage_right_x, self.stage_back_y)),
        ]

    def get_exit_goals(self, num_points=5, margin_ratio=0.1):
        """Return evenly spaced goal points along each exit, skipping edge margins."""
        goals = []
        for (x1, y1), (x2, y2) in self.exits:
            length = np.linalg.norm([x2 - x1, y2 - y1])
            margin = margin_ratio * length
            vec = np.array([x2 - x1, y2 - y1])
            unit_vec = vec / np.linalg.norm(vec)

            start = np.array([x1, y1]) + unit_vec * margin
            end = np.array([x2, y2]) - unit_vec * margin

            for i in range(num_points):
                t = i / (num_points - 1)
                point = start + t * (end - start)
                goals.append(point)
        return goals
