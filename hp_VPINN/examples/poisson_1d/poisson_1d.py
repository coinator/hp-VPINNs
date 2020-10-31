import tensorflow as tf
import numpy as np

from hp_VPINN.utilities import tf, np
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial, gauss_lobatto_jacobi_weights
from hp_VPINN.utilities.plotting import plot
from hp_VPINN.utilities.test_functions import jacobi_test_function
from hp_VPINN.vpinn.vpinn import VPINN

if __name__ == "__main__":

    learning_rate = 0.001
    optimization_iterations = 5000 + 1
    optimization_threshold = 2e-32
    variational_form = 2
    n_elements = 3
    net_layers = [1] + [20] * 4 + [1]
    test_functions_per_element = 60
    n_quadrature_points = 80
    boundary_loss_weight = 1

    omega = 8 * np.pi
    amp = 1
    r1 = 80

    def u_exact(x):
        utemp = 0.1 * np.sin(omega * x) + np.tanh(r1 * x)
        return amp * utemp

    def f(x):
        gtemp = -0.1 * (omega**2) * np.sin(omega * x) - (2 * r1**2) * (np.tanh(
            r1 * x)) / ((np.cosh(r1 * x))**2)
        return -amp * gtemp

    def test_function(n, x):
        return jacobi_test_function(n, x)

    x_quad, w_quad = gauss_lobatto_jacobi_weights(n_quadrature_points, 0, 0)

    x_l, x_r = [-1, 1]
    delta_x = (x_r - x_l) / n_elements
    grid = np.asarray([x_l + i * delta_x for i in range(n_elements + 1)])

    f_elements = []
    for e in range(n_elements):
        x_quad_element = grid[e] + (grid[e + 1] - grid[e]) / 2 * (x_quad + 1)
        jacobian = (grid[e + 1] - grid[e]) / 2
        test_functions_element = test_function(test_functions_per_element,
                                               x_quad)

        f_quad_element = f(x_quad_element)
        f_element = jacobian * np.asarray([
            sum(w_quad * f_quad_element * test_functions_element[i])
            for i in range(test_functions_per_element)
        ])
        f_element = f_element[:, None]
        f_elements.append(f_element)

    f_elements = np.asarray(f_elements)

    x_boundary = np.asarray([-1.0, 1.0])[:, None]
    u_boundary = u_exact(x_boundary)

    x_quad, w_quad = gauss_lobatto_jacobi_weights(n_quadrature_points, 0, 0)

    X_quad_train = x_quad[:, None]
    W_quad_train = w_quad[:, None]

    test_points = 2000

    x_test = np.linspace(-1, 1, test_points)
    x_prediction = x_test[:, None]
    u_correct = u_exact(x_test)[:, None]

    model = VPINN(net_layers)
    model.boundary(x_boundary, u_boundary)
    model.element_losses(X_quad_train, W_quad_train, variational_form,
                         boundary_loss_weight, f_elements, grid)
    model.optimizer(learning_rate)
    total_record = model.train(optimization_iterations, optimization_threshold,
                               [])
    u_prediction = model.predict(x_prediction)

    plot(x_quadrature=X_quad_train,
         x_boundary=x_boundary,
         x_prediction=x_prediction,
         u_prediction=u_prediction,
         u_correct=u_correct,
         total_record=total_record,
         grid=grid)
