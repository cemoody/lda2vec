import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable


def _orthogonal_matrix(shape):
    # Stolen from blocks:
    # github.com/mila-udem/blocks/blob/master/blocks/initialization.py
    M1 = np.random.randn(shape[0], shape[0])
    M2 = np.random.randn(shape[1], shape[1])

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))

    n_min = min(shape[0], shape[1])
    return np.dot(Q1[:, :n_min], Q2[:n_min, :])


class EmbedMixture(chainer.Chain):
    r""" A single document is encoded as a multinomial mixture of latent topics.
    The mixture is defined on simplex, so that mixture weights always sum
    to 100%. The latent topic vectors resemble word vectors whose elements are
    defined over all real numbers.

    For example, a single document mix may be :math:`[0.9, 0.1]`, indicating
    that it is 90% in the first topic, 10% in the second. An example topic
    vector looks like :math:`[1.5e1, -1.3e0, +3.4e0, -0.2e0]`, which is
    largely uninterpretable until you measure the words most similar to this
    topic vector.

    A single document vector :math:`\vec{e}` is composed as weights :math:`c_j`
    over topic vectors :math:`\vec{T_j}`:

    .. math::

        \vec{e}=\Sigma_{j=0}^{j=n\_topics}c_j\vec{T_j}

    This is usually paired with regularization on the weights :math:`c_j`.
    If using a Dirichlet prior with low alpha, these weights will be sparse.

    Args:
        n_documents (int): Total number of documents
        n_topics (int): Number of topics per document
        n_dim (int): Number of dimensions per topic vector (should match word
            vector size)

    Attributes:
        weights : chainer.links.EmbedID
            Unnormalized topic weights (:math:`c_j`). To normalize these
            weights, use `F.softmax(weights)`.
        factors : chainer.links.Parameter
            Topic vector matrix (:math:`T_j`)

    .. seealso:: :func:`lda2vec.dirichlet_likelihood`
    """

    def __init__(self, n_documents, n_topics, n_dim, dropout_ratio=0.2,
                 temperature=1.0):
        self.n_documents = n_documents
        self.n_topics = n_topics
        self.n_dim = n_dim
        self.dropout_ratio = dropout_ratio
        factors = _orthogonal_matrix((n_topics, n_dim)).astype('float32')
        factors /= np.sqrt(n_topics + n_dim)
        super(EmbedMixture, self).__init__(
            weights=L.EmbedID(n_documents, n_topics),
            factors=L.Parameter(factors))
        self.temperature = temperature
        self.weights.W.data[...] /= np.sqrt(n_documents + n_topics)

    def __call__(self, doc_ids, update_only_docs=False):
        """ Given an array of document integer indices, returns a vector
        for each document. The vector is composed of topic weights projected
        onto topic vectors.

        Args:
            doc_ids : chainer.Variable
                One-dimensional batch vectors of IDs

        Returns:
            doc_vector : chainer.Variable
                Batch of two-dimensional embeddings for every document.
        """
        # (batchsize, ) --> (batchsize, multinomial)
        proportions = self.proportions(doc_ids, softmax=True)
        # (batchsize, n_factors) * (n_factors, n_dim) --> (batchsize, n_dim)
        factors = F.dropout(self.factors(), ratio=self.dropout_ratio)
        if update_only_docs:
            factors.unchain_backward()
        w_sum = F.matmul(proportions, factors)
        return w_sum

    def proportions(self, doc_ids, softmax=False):
        """ Given an array of document indices, return a vector
        for each document of just the unnormalized topic weights.

        Returns:
            doc_weights : chainer.Variable
                Two dimensional topic weights of each document.
        """
        w = self.weights(doc_ids)
        if softmax:
            size = w.data.shape
            mask = self.xp.random.random_integers(0, 1, size=size)
            y = (F.softmax(w * self.temperature) *
                 Variable(mask.astype('float32')))
            norm, y = F.broadcast(F.expand_dims(F.sum(y, axis=1), 1), y)
            return y / (norm + 1e-7)
        else:
            return w
