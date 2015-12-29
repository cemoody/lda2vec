import numpy as np
import chainer.links as L
from chainer import Variable

from lda2vec import dirichlet_likelihood


def test_concentration():
    """ Test that alpha > 1.0 on a dense vector has a higher likelihood
    than alpha < 1.0 on a dense vector, and test that a sparse vector
    has the opposite character. """

    dense = np.random.randn(5, 10).astype('float32')
    sparse = np.random.randn(5, 10).astype('float32')
    sparse[:, 1:] /= 1e5
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


def test_embed():
    """ Test that embedding is treated like a Variable"""

    embed_dense = L.EmbedID(5, 10)
    embed_sparse = L.EmbedID(5, 10)
    embed_dense.W.data[:] = np.random.randn(5, 10).astype('float32')
    embed_sparse.W.data[:] = np.random.randn(5, 10).astype('float32')
    embed_sparse.W.data[:, 1:] /= 1e5
    dhl_dense_01 = dirichlet_likelihood(embed_dense, alpha=0.1).data
    dhl_sparse_01 = dirichlet_likelihood(embed_sparse, alpha=0.1).data

    msg = "Sparse vector has higher likelihood than dense with alpha=0.1"
    assert dhl_sparse_01 > dhl_dense_01, msg
