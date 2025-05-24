import numpy as np

class Environment:
    def __init__(self):
        self.width = 20
        self.height = 20

        # Define fixed exit line (door in bottom wall)
        self.exits = [((8, 0), (12, 0))]

        # Define walls (excluding the exit part)
        self.walls = [
            ((0, 0), (8, 0)),
            ((12, 0), (20, 0)),
            ((0, 20), (20, 20)),
            ((0, 0), (0, 20)),
            ((20, 0), (20, 20))
        ]

    def get_exit_goals(self, num_points=5):
        """Return a list of goal points sampled along the exit line(s)."""
        goals = []
        for (x1, y1), (x2, y2) in self.exits:
            for i in range(num_points):
                t = i / (num_points - 1)
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                goals.append(np.array([x, y]))
        return goals
