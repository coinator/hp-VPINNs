from hp_VPINN.elements.element import Element
from hp_VPINN.utilities import np, tf
from hp_VPINN.utilities.arg_parsing import results_dir
from hp_VPINN.utilities.grid import create_grid
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import (
    gauss_lobatto_jacobi_weights, jacobi_polynomial)
from hp_VPINN.utilities.plotting import plot
from hp_VPINN.utilities.test_functions import jacobi_test_function
from hp_VPINN.vpinn.vpinn import VPINN

if __name__ == "__main__":

    learning_rate = 0.001
    optimization_iterations = 700 + 1
    optimization_threshold = 2e-32
    variational_form = 2
    n_elements = 3
    net_layers = [1] + [20] * 3 + [1]
    test_functions_per_element = 60
    n_quadrature_points = 90
    boundary_loss_weight = 1
    test_points = 2000

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

    grid = create_grid(n_elements)

    elements = [
        Element(x_quad, w_quad, test_functions_per_element, test_function, f,
                grid[i], grid[i + 1]) for i in range(n_elements)
    ]

    x_boundary = np.array([-1.0, 1.0])[:, None]
    u_boundary = u_exact(x_boundary)

    x_quad_train = x_quad[:, None]
    w_quad_train = w_quad[:, None]

    x_test = np.linspace(-1, 1, test_points)
    x_prediction = x_test[:, None]
    u_correct = u_exact(x_test)[:, None]

    model = VPINN(net_layers)
    model.boundary(x_boundary, u_boundary)
    model.element_losses(x_quad_train, w_quad_train, variational_form,
                         boundary_loss_weight, elements)
    model.optimizer(learning_rate)
    model.exact(x_test, u_exact(x_test))
    total_record = model.train(optimization_iterations, optimization_threshold,
                               [])
    u_prediction = model.predict(x_prediction)

    plot(x_quadrature=x_quad_train,
         x_boundary=x_boundary,
         x_prediction=x_prediction,
         u_prediction=u_prediction,
         u_correct=u_correct,
         total_record=total_record,
         grid=grid,
         results_dir=results_dir)
