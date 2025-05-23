class Agent:
    def __init__(self, x, y, goal):
        self.position = [x, y]
        self.goal = goal
        self.has_exited = False

    def move_toward_goal(self, speed=0.2):
        if self.has_exited:
            return

        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        norm = (dx**2 + dy**2) ** 0.5
        if norm > 0:
            self.position[0] += speed * dx / norm
            self.position[1] += speed * dy / norm