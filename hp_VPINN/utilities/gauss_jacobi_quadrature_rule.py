# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi


def Jacobi(n, a, b, x):
    x = np.array(x)
    return (jacobi(n, a, b)(x))


def DJacobi(n, a, b, x, k: int):
    x = np.array(x)
    ctemp = gamma(a + b + n + 1 + k) / (2**k) / gamma(a + b + n + 1)
    return (ctemp * Jacobi(n - k, a + k, b + k, x))


def GaussJacobiWeights(Q: int, a, b):
    [X, W] = roots_jacobi(Q, a, b)
    return [X, W]


def GaussLobattoJacobiWeights(Q: int, a, b):
    W = []
    X = roots_jacobi(Q - 2, a + 1, b + 1)[0]
    if a == 0 and b == 0:
        W = 2 / ((Q - 1) * (Q) * (Jacobi(Q - 1, 0, 0, X)**2))
        Wl = 2 / ((Q - 1) * (Q) * (Jacobi(Q - 1, 0, 0, -1)**2))
        Wr = 2 / ((Q - 1) * (Q) * (Jacobi(Q - 1, 0, 0, 1)**2))
    else:
        W = 2**(a + b + 1) * gamma(a + Q) * gamma(b + Q) / (
            (Q - 1) * gamma(Q) * gamma(a + b + Q + 1) *
            (Jacobi(Q - 1, a, b, X)**2))
        Wl = (b + 1) * 2**(a + b + 1) * gamma(a + Q) * gamma(b + Q) / (
            (Q - 1) * gamma(Q) * gamma(a + b + Q + 1) *
            (Jacobi(Q - 1, a, b, -1)**2))
        Wr = (a + 1) * 2**(a + b + 1) * gamma(a + Q) * gamma(b + Q) / (
            (Q - 1) * gamma(Q) * gamma(a + b + Q + 1) *
            (Jacobi(Q - 1, a, b, 1)**2))
    W = np.append(W, Wr)
    W = np.append(Wl, W)
    X = np.append(X, 1)
    X = np.append(-1, X)
    return [X, W]
