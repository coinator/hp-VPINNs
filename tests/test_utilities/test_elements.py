import pytest

from hp_VPINN.elements.element import Element
from hp_VPINN.utilities.gauss_jacobi_quadrature_rule import gauss_lobatto_jacobi_weights
from hp_VPINN.utilities.test_functions import jacobi_test_function


@pytest.fixture()
def element(np):
    def f(x):
        return np.ones(x.shape)

    x_quad, w_quad = gauss_lobatto_jacobi_weights(3, 0, 0)
    yield Element(x_quad, w_quad, 10, jacobi_test_function, f, -0.1, 0.3)


def test_element_test_functions(np, element):
    assert element.n_test_functions == 10


def test_element_x_quad_mapped(np, element):
    # check within tolerance of 1e-08
    assert np.allclose(element.x_quad_mapped, np.array([-0.1, 0.1, 0.3]))


def test_jacobian(np, element):
    assert element.jacobian == 0.2


def test_f_shape(np, element):
    assert element.f.shape == (element.n_test_functions, 1)
