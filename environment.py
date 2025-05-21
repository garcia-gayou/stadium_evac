class Environment:
    def __init__(self, width, height, exits):
        self.width = width
        self.height = height
        self.exits = exits  # List of exit (x, y) positions
        self.obstacles = []  # Add later if needed

    def is_exit(self, position):
        return position in self.exits
