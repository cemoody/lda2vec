import numpy as np
from chainer import Variable

from ..lda2vec.dirichlet_likelihood import dirichlet_likelihood


def test_concentration():
    """ Test that alpha > 1.0 on a dense vector has a higher likelihood
    than alpha < 1.0 on a dense vector, and test that a sparse vector
    has the opposite character. """

    dense = np.abs(np.random.randn(5, 10).astype('float32'))
    dense /= dense.max(axis=0)
    sparse = np.abs(np.random.randn(5, 10).astype('float32'))
    sparse[1:, :] = 0.0
    sparse /= sparse.max(axis=0)
    weights = Variable(dense)
    dhl_dense_10 = dirichlet_likelihood(weights, alpha=10.0).data
    dhl_dense_01 = dirichlet_likelihood(weights, alpha=0.1).data
    weights = Variable(sparse)
    dhl_sparse_10 = dirichlet_likelihood(weights, alpha=10.0).data
    dhl_sparse_01 = dirichlet_likelihood(weights, alpha=0.1).data

    msg = "Sparse vector has higher likelihood than dense with alpha=0.1"
    assert dhl_sparse_01 > dhl_dense_01, msg
    msg = "Dense vector has higher likelihood than sparse with alpha=10.0"
    assert dhl_dense_10 > dhl_sparse_10, msg