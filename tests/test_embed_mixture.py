import numpy as np
from chainer import Variable

from lda2vec import EmbedMixture


def softmax(v):
    return np.exp(v) / np.sum(np.exp(v))


def test_embed_mixture():
    """ Manually test the projection logic between topic weights and vectors"""
    # Ten documents, two topics, five hidden dimensions
    em = EmbedMixture(10, 2, 5, dropout_ratio=0.0)
    doc_ids = Variable(np.arange(1, dtype='int32'))
    doc_vector = em(doc_ids).data
    # weights -- (n_topics)
    weights = softmax(em.weights.W.data[0, :])
    un_weights = softmax(em.unnormalized_weights(doc_ids).data[0, :])
    # (n_hidden) = (n_topics) . (n_topics, n_hidden)
    test = np.sum(weights * em.factors.W.data.T, axis=1)
    assert np.allclose(doc_vector, test)
    assert np.allclose(un_weights, weights)
