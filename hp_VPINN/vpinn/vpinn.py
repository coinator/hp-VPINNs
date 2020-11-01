import time

from hp_VPINN.utilities import np, tf
from hp_VPINN.utilities.nn import NN
from hp_VPINN.utilities.test_functions import jacobi_test_function, jacobi_test_function_derivative


class VPINN(NN):
    def __init__(self, layers):
        NN.__init__(self, layers)

    def boundary(self, x_boundary, u_boundary):
        self.x = x_boundary
        self.u = u_boundary
        self.x_tf = tf.constant(self.x[:, None])
        self.u_tf = tf.constant(self.u[:, None])

    def element_losses(self, x_quadrature, w_quadrature, variational_form,
                       boundary_loss_weight, elements):
        self.x_quadrature = x_quadrature
        self.w_quadrature = w_quadrature
        self.elements = elements

        self.x_quad = tf.placeholder(tf.float64,
                                     shape=[None, self.x_quadrature.shape[1]])

        self.u_nn_boundary = self.net_u(self.x_tf)

        self.x_prediction = tf.placeholder(tf.float64,
                                           shape=[None, self.x.shape[1]])
        self.u_nn_prediction = self.net_u(self.x_prediction)

        self.varloss_total = 0
        for e in elements:
            f_element = e.f
            n_test_functions = e.n_test_functions

            x_quad_element = tf.constant(e.x_quad_mapped[:, None])
            jacobian = e.jacobian

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
        return jacobi_test_function(n_test_functions, x)

    def test_function_derivative(self, n_test_functions, x):
        return jacobi_test_function_derivative(n_test_functions, x)

    def predict(self, x):
        u_pred = self.sess.run(self.u_nn_prediction, {self.x_prediction: x})
        return u_pred

    def train(self, n_iterations, treshold, total_record):

        tf_dict = {
            self.x_quad: self.x_quadrature
        }
        start_time = time.time()
        for it in range(n_iterations):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_valueb = self.sess.run(self.lossb, tf_dict)
                loss_valuev = self.sess.run(self.lossv, tf_dict)
                total_record.append(np.array([it, loss_value]))

                if loss_value < treshold:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    return total_record

            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = 'It: %d, Lossb: %.3e, Lossv: %.3e, Time: %.2f'
                print(str_print % (it, loss_valueb, loss_valuev, elapsed))
                start_time = time.time()
        return total_record
