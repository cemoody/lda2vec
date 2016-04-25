import chainer.links as L
import chainer.functions as F
from chainer import Chain

from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

import numpy as np


class LDA2TF(Chain):
    def __init__(self, n_vocab, n_docs, n_doc_topics, n_units, k=20.0,
                 dropout_ratio=0.5):
        self.k = k
        self.dropout_ratio = dropout_ratio
        em = EmbedMixture(n_docs, n_doc_topics, n_units,
                          dropout_ratio=dropout_ratio)
        kwargs = {}
        kwargs['mixture'] = em
        kwargs['embed'] = L.EmbedID(n_vocab, n_units)
        super(LDA2TF, self).__init__(**kwargs)

    def forward(self, idxx, idxy, idxd, cnt_joint, cntx, cnty, cntd,
                cnt_obs):
        # Compute loss as log ratio between expected count from marginals
        # of each token and each document prevalence
        # So expected = #(wx) #(wy) #(c) / #(obs)^3
        #    observed = #(wx, wy, c) / #(obs)
        # and then the excess should be modeled as three interactions
        # between word1, word2, and the context:
        # predicted = wx * wy + wy * context + wx* context
        # So log(observed / (k * expected)) = predicted
        # Note that hgher k here is effectively forcing us to pay attention
        # to pairs of words with high frequencies
        idxx, idxy, idxd = move(self.xp, idxx, idxy, idxd)
        cntx, cnty, cntd = move(self.xp, cntx, cnty, cntd)
        cnt_joint, = move(self.xp, cnt_joint)
        mult = np.log(self.k)
        expected = F.log(cntx) - np.log(cnt_obs)
        expected += F.log(cnty) - np.log(cnt_obs)
        expected += F.log(cntd) - np.log(cnt_obs)
        observed = F.log(cnt_joint) - np.log(cnt_obs)
        va, vb, vd = self.embed(idxx), self.embed(idxy), self.mixture(idxd)
        predicted = F.sum(va * vb, axis=1)
        predicted += F.sum(va * vd, axis=1)
        predicted += F.sum(vb * vd, axis=1)
        loss = F.mean_squared_error(predicted, observed - expected - mult)
        return loss

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture.weights)
        return dl1
