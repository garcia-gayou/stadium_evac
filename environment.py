class Environment:
    def __init__(self, width, height, exits):
        self.width = width
        self.height = height
        self.exits = exits  # list of (x, y) tuples
        self.walls = [  # define walls as segments: [(min), (max)]
            [(0, 0), (width, 0)],     # bottom
            [(0, 0), (0, height)],    # left
            [(width, 0), (width, height)],  # right
            [(0, height), (width, height)]  # top
        ]
