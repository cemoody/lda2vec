from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class LDA2Vec(Chain):
    def __init__(self, n_stories=100, n_story_topics=10,
                 n_authors=100, n_author_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.5, train=True,
                 counts=None, n_samples=15, word_dropout_ratio=0.0):
        em1 = EmbedMixture(n_stories, n_story_topics, n_units,
                           dropout_ratio=dropout_ratio)
        em2 = EmbedMixture(n_authors, n_author_topics, n_units,
                           dropout_ratio=dropout_ratio)
        kwargs = {}
        kwargs['mixture_sty'] = em1
        kwargs['mixture_aut'] = em2
        kwargs['sampler'] = L.NegativeSampling(n_units, counts, n_samples)
        super(LDA2Vec, self).__init__(**kwargs)
        rand = np.random.random(self.sampler.W.data.shape)
        self.sampler.W.data[:, :] = rand[:, :]
        self.n_units = n_units
        self.train = train
        self.dropout_ratio = dropout_ratio
        self.word_dropout_ratio = word_dropout_ratio
        self.n_samples = n_samples

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture_sty.weights)
        dl2 = dirichlet_likelihood(self.mixture_aut.weights)
        return dl1 + dl2

    def fit_partial(self, rsty_ids, raut_ids, rwrd_ids, window=5):
        sty_ids, aut_ids, wrd_ids = move(self.xp, rsty_ids, raut_ids, rwrd_ids)
        pivot_idx = next(move(self.xp, rwrd_ids[window: -window]))
        pivot = F.embed_id(pivot_idx, self.sampler.W)
        sty_at_pivot = rsty_ids[window: -window]
        aut_at_pivot = raut_ids[window: -window]
        sty = self.mixture_sty(next(move(self.xp, sty_at_pivot)))
        aut = self.mixture_aut(next(move(self.xp, aut_at_pivot)))
        loss = 0.0
        start, end = window, rwrd_ids.shape[0] - window
        context = sty + aut + F.dropout(pivot, self.dropout_ratio)
        for frame in range(-window, window + 1):
            # Skip predicting the current pivot
            if frame == 0:
                continue
            # Predict word given context and pivot word
            # The target starts before the pivot
            targetidx = rwrd_ids[start + frame: end + frame]
            sty_at_target = rsty_ids[start + frame: end + frame]
            aut_at_target = raut_ids[start + frame: end + frame]
            sty_is_same = sty_at_target == sty_at_pivot
            aut_is_same = aut_at_target == aut_at_pivot
            # Randomly dropout words (default is to never do this)
            rand = np.random.uniform(0, 1, sty_is_same.shape[0])
            mask = (rand > self.word_dropout_ratio).astype('bool')
            sty_and_aut_are_same = np.logical_and(sty_is_same, aut_is_same)
            weight = np.logical_and(sty_and_aut_are_same, mask).astype('int32')
            # If weight is 1.0 then targetidx
            # If weight is 0.0 then -1
            targetidx = targetidx * weight + -1 * (1 - weight)
            target, = move(self.xp, targetidx)
            loss = self.sampler(context, target)
            loss.backward()
        return loss.data
