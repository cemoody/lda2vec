from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood

from chainer import Variable
from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class SimpleLDA2Vec(Chain):
    def __init__(self, loss_type, n_documents=100, n_document_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.5, train=True,
                 counts=None, n_samples=5):
        assert loss_type in ['skipgram', 'neg_sample']
        kwargs = {}
        em = EmbedMixture(n_documents, n_document_topics, n_units,
                          dropout_ratio=dropout_ratio)
        kwargs['mixture1'] = em
        kwargs['embed'] = L.EmbedID(n_vocab, n_units)
        if loss_type == 'skipgram':
            kwargs['vec2word'] = L.Linear(n_units, n_vocab)
        else:
            kwargs['sampler'] = L.NegativeSampling(n_units, counts, n_samples)
        super(SimpleLDA2Vec, self).__init__(**kwargs)
        self.loss_type = loss_type
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
        context = self.mixture1(next(self.move(context_at_pivot)))
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
            if self.loss_type == 'skipgram':
                # Keep the target index if the context was the same
                # Change to -1 if the context switched
                # Cupy doesn't support boolean masking making this hard
                targetidx = targetidx * context_same - 1 * (~context_same)
                target = next(self.move(targetidx))
                logprob = self.vec2word(combined)
                loss += F.softmax_cross_entropy(logprob, target)
            else:
                target, = self.move(targetidx)
                weight, = self.move(context_same.astype('float32'))
                loss += self.neg_sample(combined, target, weight)
        return loss

    def neg_sample(self, combined, target, weight):
        batchsize = target.data.shape[0]
        # Draw negative sample word indices
        samples = self.sampler.sampler.sample((self.n_samples, batchsize))
        samples = Variable(self.xp.ravel(samples))
        # Negate the negatively-sampled contexts
        pos_neg = (combined, ) + (-combined,) * self.n_samples
        contexts = F.concat(pos_neg, axis=0)
        # Regardless of pos/neg, ignore the same set of observations
        weights = F.concat((weight, ) * (self.n_samples + 1), axis=0)
        # Get word vectors for all targets
        targets = F.concat((target, samples), axis=0)
        target_vectors = F.dropout(self.embed(targets), self.dropout_ratio)
        # Calculate the sigmoid loss
        inner = F.sum(contexts * target_vectors, axis=1)
        loss = F.sum(F.softplus(-inner * weights))
        return loss

    def most_similar(self, word_index):
        word_index, = self.move(np.array([word_index]))
        input_vector, = self.embed(word_index).data.copy()
        lib = self.embed.W.data.copy()
        similarities = lib.dot(input_vector)
        return similarities
