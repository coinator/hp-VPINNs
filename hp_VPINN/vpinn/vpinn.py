import time

from hp_VPINN.utilities import np, tf
from hp_VPINN.utilities.nn import NN
from hp_VPINN.utilities.test_functions import jacobi_test_function, jacobi_test_function_derivatives


class VPINN(NN):
    def __init__(self, layers, weights=None, biases=None):
        NN.__init__(self, layers, weights, biases)

    def boundary(self, x_boundary, u_boundary):
        self.x_boundary = tf.constant(x_boundary[:, None])
        self.u_boundary = tf.constant(u_boundary[:, None])

    def element_losses(self, x_quadrature, w_quadrature, variational_form,
                       boundary_loss_weight, elements):
        self.x_quadrature = x_quadrature
        self.w_quadrature = w_quadrature
        self.elements = elements

        self.x_quad = tf.placeholder(tf.float64,
                                     shape=[None, self.x_quadrature.shape[1]])

        self.u_nn_boundary = self.net_u(self.x_boundary)

        self.x_prediction = tf.placeholder(
            tf.float64, shape=[None, self.x_boundary.shape[1]])
        self.u_nn_prediction = self.net_u(self.x_prediction)

        self.varloss_total = 0
        for e in elements:
            f_element = e.f
            n_test_functions = e.n_test_functions

            x_quad_element = tf.constant(e.x_quad_mapped[:, None])
            jacobian = e.jacobian

            test_quad_element = self.test_function(n_test_functions,
                                                   self.x_quadrature)
            d1test_quad_element, d2test_quad_element = self.test_function_derivatives(
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

        self.loss_boundary = tf.reduce_mean(
            tf.square(self.u_boundary - self.u_nn_boundary))
        self.loss = boundary_loss_weight * self.loss_boundary + self.varloss_total

    def optimizer(self, learning_rate):
        self.optimizer_adam = tf.train.AdamOptimizer(learning_rate)
        self.train_op_adam = self.optimizer_adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def net_du(self, x):
        u = self.net_u(x)
        d1u = tf.gradients(u, x)[0]
        d2u = tf.gradients(d1u, x)[0]
        return d1u, d2u

    def test_function(self, n_test_functions, x):
        return jacobi_test_function(n_test_functions, x)

    def test_function_derivatives(self, n_test_functions, x):
        return jacobi_test_function_derivatives(n_test_functions, x)

    def predict(self, x):
        return self.sess.run(self.u_nn_prediction, {self.x_prediction: x})

    def exact(self, x_test, u_exact):
        self.u_exact = tf.constant(u_exact[:, None])
        self.x_test = x_test[:, None]
        self.l2_error = tf.reduce_mean(
            tf.square(self.u_nn_prediction - self.u_exact))

    def train(self, n_iterations, treshold, total_record):
        start_time = time.time()
        for it in range(n_iterations):
            self.sess.run(self.train_op_adam)

            if it % 10 == 0:
                loss_value = self.sess.run(self.loss)
                loss_valueb = self.sess.run(self.loss_boundary)
                loss_valuev = self.sess.run(self.varloss_total)
                l2_errorv = self.sess.run(self.l2_error,
                                          {self.x_prediction: self.x_test})
                total_record.append(np.array([it, loss_value, l2_errorv]))

                if loss_value < treshold:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    return total_record

            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = 'It: %d, Lossb: %.3e, Lossv: %.3e, ErrL2: %.3e, Time: %.2f'
                print(str_print %
                      (it, loss_valueb, loss_valuev, l2_errorv, elapsed))
                start_time = time.time()
        return total_record
