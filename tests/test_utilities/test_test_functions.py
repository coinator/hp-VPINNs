import pytest

from hp_VPINN.utilities.test_functions import jacobi_test_function, jacobi_test_function_derivatives


@pytest.fixture
def test_function(np):
    x = np.linspace(-1, 1, 10)
    test_function = jacobi_test_function(5, x)
    yield test_function


@pytest.fixture
def test_function_derivatives(np):
    x = np.linspace(-1, 1, 10)
    test_function_derivative = jacobi_test_function_derivatives(5, x)
    yield test_function_derivative


def test_jacobi_test_function_dimension(test_function):
    assert len(test_function) == 5
    for test_f in test_function:
        assert len(test_f) == 10


def test_jacobi_test_function_compactness(test_function):
    for test_f in test_function:
        assert test_f[0] == 0
        assert test_f[-1] == 0


def test_jacobi_test_function_derivative_dimension(test_function_derivatives):
    assert len(test_function_derivatives) == 2
    for derivative in test_function_derivatives:
        assert len(derivative) == 5
        for test_function_d in derivative:
            assert len(test_function_d) == 10
