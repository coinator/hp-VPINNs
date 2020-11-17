from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import jacobi_polynomial, jacobi_polynomial_derivative, gauss_lobatto_jacobi_weights


# use the np fixtures, for randomness' sake
def test_jacobi_polynomial_even_degree(np):
    x = np.linspace(-1, 1, 10)
    polynomial = jacobi_polynomial(10, 0, 0, x)
    assert polynomial[0] == polynomial[-1]
    assert polynomial[0] == 1.0


def test_jacobi_polynomial_odd_degree(np):
    x = np.linspace(-1, 1, 10)
    polynomial = jacobi_polynomial(11, 0, 0, x)
    assert polynomial[0] == -polynomial[-1]
    assert polynomial[0] == -1.0


def test_gauss_lobatto_weights_basic_case(np):
    x, w = gauss_lobatto_jacobi_weights(3, 0, 0)
    assert np.array_equal(x, np.array([-1, 0, 1]))
    assert np.array_equal(w, np.array([1.0 / 3, 4.0 / 3, 1.0 / 3]))


def test_gauss_lobatto_weights_positive_weights(np):
    x, w = gauss_lobatto_jacobi_weights(95, 0, 0)
    assert (w > 0).all()
