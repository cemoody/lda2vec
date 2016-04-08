from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class LDA2LSTM(Chain):
    def __init__(self, n_documents=100, n_document_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.0, train=True):
        em = EmbedMixture(n_documents, n_document_topics, n_units * 2,
                          dropout_ratio=dropout_ratio)
        kwargs = {}
        kwargs['mixture'] = em
        kwargs['decoder'] = L.LSTM(n_units, n_units)
        kwargs['vec2word'] = L.Linear(n_units, n_vocab)
        super(LDA2LSTM, self).__init__(**kwargs)
        self.n_units = n_units
        self.train = train
        self.dropout_ratio = dropout_ratio

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture.weights)
        return dl1

    def variation(self, x):
        mu, ln_var = F.split_axis(x, 2, 1)
        z = F.gaussian(mu, ln_var)
        kl = F.gaussian_kl_divergence(mu, ln_var)
        return kl, z

    def fit_partial(self, rdoc_ids, rwrd_idx):
        rempty = np.zeros_like(rwrd_idx).astype('float32')
        doc_ids, empty = move(self.xp, rdoc_ids, rempty)
        mean_sigma = self.mixture(doc_ids)
        kl, doc = self.variation(mean_sigma)
        self.decoder.reset_state()
        self.decoder.h = doc
        pred = F.dropout(self.vec2word(self.decoder(empty)))
        loss = None
        for frame in range(rdoc_ids.shape[1]):
            target, = move(self.xp, rwrd_idx[:, frame])
            l = F.softmax_cross_entropy(pred, target)
            loss = l if loss is None else l + loss
            pred = F.dropout(self.vec2word(self.decoder(target)))
        return loss, kl
