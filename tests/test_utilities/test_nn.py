import pytest

from hp_VPINN.utilities.nn import NN


@pytest.fixture
def nn():
    layers = [1] + [20] * 3 + [1]
    return NN(layers)


def test_init(nn):
    assert len(nn.weights) == 4
    assert len(nn.biases) == 4
