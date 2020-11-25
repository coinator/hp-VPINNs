import pytest

from hp_VPINN.utilities.test_functions import jacobi_test_function, jacobi_test_function_derivatives

num_quadrature_points = 80
num_test_functions = 50


@pytest.fixture
def test_function(np):
    x = np.linspace(-1, 1, num_quadrature_points)
    test_function = jacobi_test_function(num_test_functions, x)
    yield test_function


@pytest.fixture
def test_function_derivatives(np):
    x = np.linspace(-1, 1, num_quadrature_points)
    test_function_derivative = jacobi_test_function_derivatives(
        num_test_functions, x)
    yield test_function_derivative


def test_jacobi_test_function_dimension(test_function):
    assert len(test_function) == num_test_functions
    for test_f in test_function:
        assert len(test_f) == num_quadrature_points


def test_jacobi_test_function_compact_support(np, test_function):
    """A proper test function is zero on the boundary and positive within"""
    for test_f in test_function:
        assert test_f[0] == 0
        assert test_f[-1] == 0
        assert test_f[1:-1].all() > 0


def test_jacobi_test_function_derivative_dimension(test_function_derivatives):
    assert len(test_function_derivatives) == 2
    for derivative in test_function_derivatives:
        assert len(derivative) == num_test_functions
        for test_function_d in derivative:
            assert len(test_function_d) == num_quadrature_points
