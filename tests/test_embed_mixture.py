import numpy as np
from chainer import Variable

from ..lda2vec.embed_mixture import EmbedMixture


def softmax(v):
    return np.exp(v) / np.sum(np.exp(v))


def test_embed_mixture():
    """ Manually test """
    # Ten documents, two topics, five hidden dimensions
    em = EmbedMixture(10, 2, 5)
    doc_ids = Variable(np.arange(1, dtype='int32'))
    doc_vector = em(doc_ids)
    # weights -- (n_topics)
    weights = softmax(em.weights.W.data[0, :])
    # (n_hidden) = (n_topics) . (n_topics, n_hidden)
    doc_vector_test = weights.T * em.factors.W.data
    assert np.all_close(doc_vector, doc_vector_test)
