# environment.py
import numpy as np

class Environment:
    def __init__(self):
        self.width = 40
        self.height = 40

        # Define multiple exits (e.g., bottom center, top left, top right)
        self.exits = [
            ((18, 0), (22, 0)),     # bottom center
            ((0, 30), (0, 34)),     # left middle
            ((40, 30), (40, 34)),   # right middle
        ]

        # Correct walls for Estadio GNP style layout
        self.walls = [
            # Bottom wall (with central exit gap)
            ((0, 0), (18, 0)),
            ((22, 0), (40, 0)),

            # Left wall (with middle exit gap)
            ((0, 0), (0, 30)),
            ((0, 34), (0, 40)),

            # Right wall (with middle exit gap)
            ((40, 0), (40, 30)),
            ((40, 34), (40, 40)),

            # Top wall
            ((0, 40), (40, 40)),

            # Middle horizontal divider
            ((0, 20), (40, 20)),

            # Scenario walls (at the top center)
            ((15, 40), (15, 35)),
            ((15, 35), (25, 35)),
            ((25, 35), (25, 40)),


        ]
    
    def get_exit_goals(self, num_points=5, margin_ratio=0.1):
        """Return evenly spaced goal points along each exit line, skipping edge margins."""
        goals = []
        for (x1, y1), (x2, y2) in self.exits:
            length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
            margin = margin_ratio * length

            vec = np.array([x2 - x1, y2 - y1])
            vec_unit = vec / np.linalg.norm(vec)

            start = np.array([x1, y1]) + vec_unit * margin
            end = np.array([x2, y2]) - vec_unit * margin

            for i in range(num_points):
                t = i / (num_points - 1)
                point = start + t * (end - start)
                goals.append(point)
        return goals
