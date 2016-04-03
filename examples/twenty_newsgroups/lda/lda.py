import chainer
import chainer.links as L
import chainer.functions as F

from lda2vec import utils, dirichlet_likelihood

import numpy as np


class LDA(chainer.Chain):
    def __init__(self, n_docs, n_topics, n_dim, n_vocab):
        factors = np.random.random((n_topics, n_dim)).astype('float32')
        super(LDA, self).__init__(proportions=L.EmbedID(n_docs, n_topics),
                                  factors=L.Parameter(factors),
                                  embedding=L.Linear(n_dim, n_vocab))
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_vocab = n_vocab
        self.n_dim = n_dim

    def forward(self, ids, bow):
        bow, ids = utils.move(self.xp, bow, ids)
        proportions = self.proportions(ids)
        ld = dirichlet_likelihood(proportions)
        doc = F.matmul(F.softmax(proportions), self.factors())
        logp = F.dropout(self.embedding(doc))
        # loss = -F.sum(bow * F.log_softmax(logp))
        sources, targets, counts = [], [], []
        lpi =  F.sum(bow * F.log_softmax(logp), axis=1)
        loss = -F.sum(lpi)
        return loss, ld
