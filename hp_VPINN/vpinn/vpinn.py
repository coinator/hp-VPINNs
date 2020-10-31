import time

from hp_VPINN.utilities import np, tf
from hp_VPINN.utilities.nn import NN
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial


class VPINN(NN):
    def __init__(self, layers):
        NN.__init__(self, layers)

    def boundary(self, x_boundary, u_boundary):
        self.x = x_boundary
        self.u = u_boundary
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])

    def element_losses(self, x_quadrature, w_quadrature, variational_form,
                       boundary_loss_weight, f_elements, grid):
        self.x_quadrature = x_quadrature
        self.w_quadrature = w_quadrature
        self.f_elements = f_elements

        self.x_quad = tf.placeholder(tf.float64,
                                     shape=[None, self.x_quadrature.shape[1]])

        self.u_nn_boundary = self.net_u(self.x_tf)

        self.x_prediction = tf.placeholder(tf.float64,
                                           shape=[None, self.x.shape[1]])
        self.u_nn_prediction = self.net_u(self.x_prediction)

        self.varloss_total = 0
        for e in range(np.shape(self.f_elements)[0]):
            f_element = self.f_elements[e]
            n_test_functions = np.shape(f_element)[0]

            x_quad_element = tf.constant(grid[e] +
                                         (grid[e + 1] - grid[e]) / 2 *
                                         (self.x_quadrature + 1))
            jacobian = (grid[e + 1] - grid[e]) / 2

            test_quad_element = self.test_function(n_test_functions,
                                                   self.x_quadrature)
            d1test_quad_element, d2test_quad_element = self.test_function_derivative(
                n_test_functions, self.x_quadrature)
            u_nn_quad_element = self.net_u(x_quad_element)
            d1u_nn_quad_element, d2u_nn_quad_element = self.net_du(
                x_quad_element)

            if variational_form == 1:
                u_nn_element = tf.reshape(
                    tf.stack([
                        -jacobian *
                        tf.reduce_sum(self.w_quadrature * d2u_nn_quad_element *
                                      test_quad_element[i])
                        for i in range(n_test_functions)
                    ]), (-1, 1))

            if variational_form == 2:
                u_nn_element = tf.reshape(
                    tf.stack([
                        tf.reduce_sum(self.w_quadrature * d1u_nn_quad_element *
                                      d1test_quad_element[i])
                        for i in range(n_test_functions)
                    ]), (-1, 1))

            residual_nn_element = u_nn_element - f_element
            loss_element = tf.reduce_mean(tf.square(residual_nn_element))
            self.varloss_total += loss_element

        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_nn_boundary))
        self.lossv = self.varloss_total
        self.loss = boundary_loss_weight * self.lossb + self.lossv

    def optimizer(self, learning_rate):
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
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
        u_pred = self.sess.run(self.u_nn_prediction, {self.x_prediction: x})
        return u_pred

    def train(self, nIter, tresh, total_record):

        tf_dict = {
            self.x_tf: self.x,
            self.u_tf: self.u,
            self.x_quad: self.x_quadrature,
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
