from hp_VPINN.utilities import np
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial


def jacobi_test_function(n_test_functions, x):
    test_total = [
        jacobi_polynomial(n + 1, 0, 0, x) - jacobi_polynomial(n - 1, 0, 0, x)
        for n in range(1, n_test_functions + 1)
    ]
    return np.array(test_total)


def jacobi_test_function_derivatives(n_test_functions, x):
    d1test_total = [((1 + 2) / 2) * jacobi_polynomial(1, 1, 1, x),
                    ((2 + 2) / 2) * jacobi_polynomial(2, 1, 1, x) -
                    ((2) / 2) * jacobi_polynomial(2 - 2, 1, 1, x)]
    d2test_total = [
        ((1 + 2) * (1 + 3) / (2 * 2)) * jacobi_polynomial(0, 2, 2, x),
        ((2 + 2) * (2 + 3) / (2 * 2)) * jacobi_polynomial(1, 2, 2, x)
    ]
    for n in range(3, n_test_functions + 1):
        d1test = ((n + 2) / 2) * jacobi_polynomial(n, 1, 1, x) - (
            (n) / 2) * jacobi_polynomial(n - 2, 1, 1, x)
        d2test = ((n + 2) * (n + 3) /
                  (2 * 2)) * jacobi_polynomial(n - 1, 2, 2, x) - (
                      (n) * (n + 1) /
                      (2 * 2)) * jacobi_polynomial(n - 3, 2, 2, x)
        d1test_total.append(d1test)
        d2test_total.append(d2test)
    return np.array(d1test_total), np.array(d2test_total)
