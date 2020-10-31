import tensorflow as tf
import numpy as np
import time

from hp_VPINN.utilities import tf, np
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial, gauss_lobatto_jacobi_weights
from hp_VPINN.utilities.plotting import plot
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
        test = jacobi_polynomial(n + 1, 0, 0, x) - jacobi_polynomial(
            n - 1, 0, 0, x)
        return test

    [x_quad, w_quad] = gauss_lobatto_jacobi_weights(n_quadrature_points, 0, 0)

    [x_l, x_r] = [-1, 1]
    delta_x = (x_r - x_l) / n_elements
    grid = np.asarray([x_l + i * delta_x for i in range(n_elements + 1)])

    F_exact_total = []
    for e in range(n_elements):
        x_quad_element = grid[e] + (grid[e + 1] - grid[e]) / 2 * (x_quad + 1)
        jacobian = (grid[e + 1] - grid[e]) / 2
        testfcn_element = np.asarray([
            test_function(n, x_quad)
            for n in range(1, test_functions_per_element + 1)
        ])

        f_quad_element = f(x_quad_element)
        F_exact_element = jacobian * np.asarray([
            sum(w_quad * f_quad_element * testfcn_element[i])
            for i in range(test_functions_per_element)
        ])
        F_exact_element = F_exact_element[:, None]
        F_exact_total.append(F_exact_element)

    F_exact_total = np.asarray(F_exact_total)

    x_boundary = np.asarray([-1.0, 1.0])[:, None]
    u_boundary = u_exact(x_boundary)

    [x_quad, w_quad] = gauss_lobatto_jacobi_weights(n_quadrature_points, 0, 0)

    X_quad_train = x_quad[:, None]
    W_quad_train = w_quad[:, None]

    test_points = 2000
    xtest = np.linspace(-1, 1, test_points)
    data_temp = np.asarray([[xtest[i], u_exact(xtest[i])]
                            for i in range(len(xtest))])
    x_prediction = data_temp.flatten()[0::2]
    u_correct = data_temp.flatten()[1::2]
    x_prediction = x_prediction[:, None]
    u_correct = u_correct[:, None]

    model = VPINN(net_layers)
    model.boundary(x_boundary, u_boundary)
    model.element_losses(X_quad_train, W_quad_train, variational_form,
                         boundary_loss_weight, F_exact_total, grid)
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
