from hp_VPINN.utilities import np, tf


def create_grid(n_elements, x_left=-1, x_right=1):
    delta_x = (x_right - x_left) / n_elements
    return np.array([x_left + i * delta_x for i in range(n_elements + 1)])
