from hp_VPINN.utilities import np, tf
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial


class NN:
    def __init__(self, layers, weights=None, biases=None):
        if not (weights and biases):
            self.weights, self.biases = self.initialize_NN(layers)
        else:
            self.weights = weights
            self.biases = biases

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64),
                            dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim), dtype=np.float64)
        return tf.Variable(tf.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev,
                                               dtype=tf.float64),
                           dtype=tf.float64)

    def neural_net(self, x, weights, biases):
        out = x
        for w, b in zip(weights[:-1], biases[:-1]):
            out = tf.sin(tf.add(tf.matmul(out, w), b))
        w = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(out, w), b)
        return Y

    def net_u(self, x):
        u = self.neural_net(tf.concat([x], 1), self.weights, self.biases)
        return u
