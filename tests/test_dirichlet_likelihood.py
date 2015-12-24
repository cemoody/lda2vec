import numpy as np
from chainer import Variable

from lda2vec import dirichlet_likelihood


def test_concentration():
    """ Test that alpha > 1.0 on a dense vector has a higher likelihood
    than alpha < 1.0 on a dense vector, and test that a sparse vector
    has the opposite character. """

    dense = np.abs(np.random.randn(5, 10, dtype='float32'))
    dense /= dense.max(axis=0)
    weights = Variable(dense)
    dhl_likely = dirichlet_likelihood(weights, alpha=10.0)
    dhl_unlikely = dirichlet_likelihood(weights, alpha=0.1)

    assert dhl_likely > dhl_unlikely

    sparse = np.abs(np.random.randn(5, 10, dtype='float32'))
    sparse[1:, :] = 0.0
    sparse /= sparse.max(axis=0)
    weights = Variable(sparse)
    dhl_unlikely = dirichlet_likelihood(weights, alpha=10.0)
    dhl_likely = dirichlet_likelihood(weights, alpha=0.1)

    assert dhl_likely > dhl_unlikely
