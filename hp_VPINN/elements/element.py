from hp_VPINN.utilities import np


class Element:
    def __init__(self, x_quad, w_quad, n_test_functions, test_function, f,
                 grid_left, grid_right):
        self.n_test_functions = n_test_functions

        self.x_quad_mapped = grid_left + (grid_right -
                                          grid_left) / 2 * (x_quad + 1)
        self.test_functions = test_function(self.n_test_functions, x_quad)

        f_quad_element = f(self.x_quad_mapped)

        self.jacobian = (grid_right - grid_left) / 2

        self.f = self.jacobian * np.asarray(
            [sum(w_quad * f_quad_element * t) for t in self.test_functions])
        self.f = self.f[:, None]
