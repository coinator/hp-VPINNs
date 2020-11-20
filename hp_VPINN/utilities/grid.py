from hp_VPINN.utilities import np, tf


def create_grid(n_elements):
    x_l, x_r = [-1, 1]
    delta_x = (x_r - x_l) / n_elements
    return np.array([x_l + i * delta_x for i in range(n_elements + 1)])
