import numpy as np

def generate_agent_positions(env, num_agents_bottom=100, num_agents_top=100):
    agents = []

    def density_y_front_weighted(y, y_min, y_max):
        """Higher density near the lower end of the section (closer to the divider or stage)."""
        center = y_min + (y_max - y_min) * 0.6
        return 1 / (1 + np.exp(-0.1 * (y - center)))  # sigmoid-like shape

    def is_in_stage(x, y):
        return (env.stage_left_x <= x <= env.stage_right_x) and (env.stage_front_y <= y <= env.stage_back_y)

    def rejection_sample(n, x_range, y_range, density_func, mask_func=None, max_attempts=10000):
        samples = []
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            r = np.random.rand()
            if r < density_func(y) and (mask_func is None or not mask_func(x, y)):
                samples.append((x, y))
            attempts += 1
        return samples

    # Bottom section
    bottom_samples = rejection_sample(
        num_agents_bottom,
        x_range=(0, env.width),
        y_range=(0, env.divider_y),
        density_func=lambda y: density_y_front_weighted(y, 0, env.divider_y)
    )

    # Top section, with stage avoidance
    top_samples = rejection_sample(
        num_agents_top,
        x_range=(0, env.width),
        y_range=(env.divider_y, env.height),
        density_func=lambda y: density_y_front_weighted(y, env.divider_y, env.height),
        mask_func=is_in_stage
    )

    agents.extend(bottom_samples + top_samples)
    return agents
