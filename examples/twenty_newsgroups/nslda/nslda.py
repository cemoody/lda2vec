import chainer
import chainer.links as L
import chainer.functions as F

from lda2vec import utils, dirichlet_likelihood

import numpy as np


class NSLDA(chainer.Chain):
    def __init__(self, counts, n_docs, n_topics, n_dim, n_vocab, n_samples=5):
        factors = np.random.random((n_topics, n_dim)).astype('float32')
        loss_func = L.NegativeSampling(n_dim, counts, n_samples)
        loss_func.W.data[:, :] = np.random.randn(*loss_func.W.data.shape)
        loss_func.W.data[:, :] /= np.sqrt(np.prod(loss_func.W.data.shape))
        super(NSLDA, self).__init__(proportions=L.EmbedID(n_docs, n_topics),
                                    factors=L.Parameter(factors),
                                    loss_func=loss_func)
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_vocab = n_vocab
        self.n_dim = n_dim

    def forward(self, doc, wrd, window=5):
        doc, wrd = utils.move(self.xp, doc, wrd)
        proportions = self.proportions(doc)
        ld = dirichlet_likelihood(self.proportions.W)
        context = F.matmul(F.softmax(proportions), self.factors())
        loss = self.loss_func(context, wrd)
        return loss, ld
