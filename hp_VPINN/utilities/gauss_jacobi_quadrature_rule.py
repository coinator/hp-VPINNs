# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi


def jacobi_polynomial(n, a, b, x):
    return jacobi(n, a, b)(x)


def jacobi_polynomial_derivative(n, a, b, x, k):
    " return derivative of oder k "
    ctemp = gamma(a + b + n + 1 + k) / (2**k) / gamma(a + b + n + 1)
    return ctemp * jacobi(n - k, a + k, b + k)(x)


def gauss_lobatto_jacobi_weights(n, a, b):
    X = roots_jacobi(n - 2, a + 1, b + 1)[0]

    Wl = (b + 1) * 2**(a + b + 1) * gamma(a + n) * gamma(b + n) / (
        (n - 1) * gamma(n) * gamma(a + b + n + 1) *
        (jacobi_polynomial(n - 1, a, b, -1)**2))

    W = 2**(a + b + 1) * gamma(a + n) * gamma(b + n) / (
        (n - 1) * gamma(n) * gamma(a + b + n + 1) *
        (jacobi_polynomial(n - 1, a, b, X)**2))
    
    Wr = (a + 1) * 2**(a + b + 1) * gamma(a + n) * gamma(b + n) / (
        (n - 1) * gamma(n) * gamma(a + b + n + 1) *
        (jacobi_polynomial(n - 1, a, b, 1)**2))

    W = np.append(W, Wr)
    W = np.append(Wl, W)
    X = np.append(-1, X)
    X = np.append(X, 1)
    return [X, W]
