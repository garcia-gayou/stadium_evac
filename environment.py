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

    def get_exit_goals(self, num_points=5, margin_ratio=0.05):
        """Return evenly spaced goal points along each exit line, skipping edge margins."""
        goals = []
        for (x1, y1), (x2, y2) in self.exits:
            length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
            margin = margin_ratio * length

            # Adjust start and end points to leave margins
            vec = np.array([x2 - x1, y2 - y1])
            vec_unit = vec / np.linalg.norm(vec)

            start = np.array([x1, y1]) + vec_unit * margin
            end = np.array([x2, y2]) - vec_unit * margin

            for i in range(num_points):
                t = i / (num_points - 1)
                point = start + t * (end - start)
                goals.append(point)
        return goals