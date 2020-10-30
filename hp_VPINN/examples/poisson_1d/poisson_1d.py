import tensorflow as tf
import numpy as np
import time

from hp_VPINN.utilities import tf, np
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial, gauss_lobatto_jacobi_weights
from hp_VPINN.utilities.nn import NN
from hp_VPINN.utilities.plotting import plot


class VPINN(NN):
    def __init__(self, x_boundary, u_boundary, x_quadrature, w_quadrature,
                 F_exact_total, grid, layers):

        NN.__init__(self, layers)
        self.x = x_boundary
        self.u = u_boundary
        self.xquad = x_quadrature
        self.wquad = w_quadrature
        self.F_exact_total = F_exact_total
        self.n_element = np.shape(self.F_exact_total)[0]

        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])
        self.x_quad = tf.placeholder(tf.float64,
                                     shape=[None, self.xquad.shape[1]])

        self.u_NN_boundary = self.net_u(self.x_tf)

        self.x_prediction = tf.placeholder(tf.float64,
                                           shape=[None, self.x.shape[1]])
        self.u_NN_prediction = self.net_u(self.x_prediction)

        self.varloss_total = 0
        for e in range(self.n_element):
            F_exact_element = self.F_exact_total[e]
            Ntest_element = np.shape(F_exact_element)[0]

            x_quad_element = tf.constant(grid[e] +
                                         (grid[e + 1] - grid[e]) / 2 *
                                         (self.xquad + 1))
            jacobian = (grid[e + 1] - grid[e]) / 2

            test_quad_element = self.test_function(Ntest_element, self.xquad)
            d1test_quad_element, d2test_quad_element = self.test_function_derivative(
                Ntest_element, self.xquad)
            u_NN_quad_element = self.net_u(x_quad_element)
            d1u_NN_quad_element, d2u_NN_quad_element = self.net_du(
                x_quad_element)

            if variational_form == 1:
                U_NN_element = tf.reshape(
                    tf.stack([
                        -jacobian *
                        tf.reduce_sum(self.wquad * d2u_NN_quad_element *
                                      test_quad_element[i])
                        for i in range(Ntest_element)
                    ]), (-1, 1))

            if variational_form == 2:
                U_NN_element = tf.reshape(
                    tf.stack([
                        tf.reduce_sum(self.wquad * d1u_NN_quad_element *
                                      d1test_quad_element[i])
                        for i in range(Ntest_element)
                    ]), (-1, 1))

            Res_NN_element = U_NN_element - F_exact_element
            loss_element = tf.reduce_mean(tf.square(Res_NN_element))
            self.varloss_total = self.varloss_total + loss_element

        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_NN_boundary))
        self.lossv = self.varloss_total
        self.loss = boundary_loss_weight * self.lossb + self.lossv

        self.learning_rate = learning_rate
        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def net_du(self, x):
        u = self.net_u(x)
        d1u = tf.gradients(u, x)[0]
        d2u = tf.gradients(d1u, x)[0]
        return d1u, d2u

    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = -u_xx
        return f

    def test_function(self, n_test_functions, x):
        test_total = []
        for n in range(1, n_test_functions + 1):
            test = jacobi_polynomial(n + 1, 0, 0, x) - jacobi_polynomial(
                n - 1, 0, 0, x)
            test_total.append(test)
        return np.asarray(test_total)

    def test_function_derivative(self, n_test_functions, x):
        d1test_total = []
        d2test_total = []
        for n in range(1, n_test_functions + 1):
            if n == 1:
                d1test = ((n + 2) / 2) * jacobi_polynomial(n, 1, 1, x)
                d2test = ((n + 2) * (n + 3) /
                          (2 * 2)) * jacobi_polynomial(n - 1, 2, 2, x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n == 2:
                d1test = ((n + 2) / 2) * jacobi_polynomial(n, 1, 1, x) - (
                    (n) / 2) * jacobi_polynomial(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) /
                          (2 * 2)) * jacobi_polynomial(n - 1, 2, 2, x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            else:
                d1test = ((n + 2) / 2) * jacobi_polynomial(n, 1, 1, x) - (
                    (n) / 2) * jacobi_polynomial(n - 2, 1, 1, x)
                d2test = ((n + 2) * (n + 3) /
                          (2 * 2)) * jacobi_polynomial(n - 1, 2, 2, x) - (
                              (n) * (n + 1) /
                              (2 * 2)) * jacobi_polynomial(n - 3, 2, 2, x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
        return np.asarray(d1test_total), np.asarray(d2test_total)

    def predict(self, x):
        u_pred = self.sess.run(self.u_NN_prediction, {self.x_prediction: x})
        return u_pred

    def train(self, nIter, tresh, total_record):

        tf_dict = {
            self.x_tf: self.x,
            self.u_tf: self.u,
            self.x_quad: self.xquad,
        }
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_valueb = self.sess.run(self.lossb, tf_dict)
                loss_valuev = self.sess.run(self.lossv, tf_dict)
                total_record.append(np.array([it, loss_value]))

                if loss_value < tresh:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    return total_record

            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = 'It: %d, Lossb: %.3e, Lossv: %.3e, Time: %.2f'
                print(str_print % (it, loss_valueb, loss_valuev, elapsed))
                start_time = time.time()
        return total_record



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

    model = VPINN(x_boundary, u_boundary, X_quad_train, W_quad_train,
                  F_exact_total, grid, net_layers)
    total_record = model.train(optimization_iterations, optimization_threshold, [])
    u_prediction = model.predict(x_prediction)

    plot(x_quadrature=X_quad_train,
         x_boundary=x_boundary,
         x_prediction=x_prediction,
         u_prediction=u_prediction,
         u_correct=u_correct,
         total_record=total_record,
         grid=grid)
