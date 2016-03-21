from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood

from chainer import Variable
from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class SimpleLDA2Vec(Chain):
    def __init__(self, n_documents=100, n_document_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.5, train=True,
                 counts=None, n_samples=5):
        kwargs = {}
        em = EmbedMixture(n_documents, n_document_topics, n_units,
                          dropout_ratio=dropout_ratio)
        kwargs['mixture'] = em
        kwargs['embed'] = L.EmbedID(n_vocab, n_units)
        kwargs['sampler'] = L.NegativeSampling(n_units, counts, n_samples)
        super(SimpleLDA2Vec, self).__init__(**kwargs)
        self.n_units = n_units
        self.train = train
        self.dropout_ratio = dropout_ratio
        self.n_samples = n_samples

    def move(self, *args):
        for arg in args:
            if 'float' in str(arg.dtype):
                yield Variable(self.xp.asarray(arg, dtype='float32'))
            else:
                assert 'int' in str(arg.dtype)
                yield Variable(self.xp.asarray(arg, dtype='int32'))

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture1.weights)
        return dl1

    def fit_pivot(self, rdoc_ids, rword_indices, window=5):
        # From empty token but document-initialized state predict 1st token
        doc_ids, word_indices = self.move(rdoc_ids, rword_indices)
        pivot = self.embed(next(self.move(rword_indices[window: -window])))
        context_at_pivot = rdoc_ids[window: -window]
        context = self.mixture(next(self.move(context_at_pivot)))
        loss = 0.0
        start, end = window, rword_indices.shape[0] - window
        for frame in range(-window, window):
            # Skip predicting the current pivot
            if frame == 0:
                continue
            # Predict word given context and pivot word
            combined = (F.dropout(context, self.dropout_ratio) +
                        F.dropout(pivot, self.dropout_ratio))
            # The target starts before the pivot
            targetidx = rword_indices[start + frame: end + frame]
            context_at_target = rdoc_ids[start + frame: end + frame]
            context_same = context_at_target == context_at_pivot
            targetidx[~context_same] = -1
            target, = self.move(targetidx)
            loss += self.sampler(combined, target)
        return loss

    def most_similar(self, word_index):
        word_index, = self.move(np.array([word_index]))
        input_vector, = self.embed(word_index).data.copy()
        lib = self.embed.W.data.copy()
        similarities = lib.dot(input_vector)
        return similarities
