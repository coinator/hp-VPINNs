import pytest 

from hp_VPINN.utilities import np as numpy
from hp_VPINN.utilities import tf as tensorflow

tensorflow.set_random_seed(0)
numpy.random.seed(0)

@pytest.fixture
def tf():
    return tensorflow

@pytest.fixture
def np():
    return numpy
