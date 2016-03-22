import chainer
import chainer.links as L
import chainer.functions as F

from lda2vec import utils


class NVDM(chainer.Chain):
    def __init__(self, n_vocab, n_dim):
        super(NVDM, self).__init__(l1=L.Linear(n_vocab, n_dim),
                                   l2=L.Linear(n_dim, n_dim),
                                   mu=L.Linear(n_dim, n_dim),
                                   log_sigma=L.Linear(n_dim, n_dim),
                                   embedding=L.Linear(n_dim, n_vocab))
        self.n_vocab = n_vocab
        self.n_dim = n_dim

    def encode(self, bow):
        """ Convert the bag of words vector of shape (n_docs, n_vocab)
        into latent mean log variance vectors.
        """
        lam = F.relu(self.l1(bow))
        pi = F.relu(self.l2(lam))
        mu = self.mu(pi)
        log_sigma = self.log_sigma(pi)
        return mu, log_sigma

    def decode(self, mu, log_sigma, word_compact):
        """ Decode latent document vectors into word indices of shape
        (n_docs, doc_length).
        """
        batchsize = word_compact.shape[0]
        e = self.xp.random.normal(size=(batchsize, self.n_dim))
        h = mu + F.exp(0.5 * log_sigma) * e
        log_prob = self.embedding(h)
        loss_rec = F.softmax_cross_entropy(log_prob, word_compact)
        loss_kl = F.gaussian_kl_divergence(mu, log_sigma)
        return loss_rec, loss_kl

    def fit(self, bow, word_compact):
        bow, word_compact = utils.move(self.xp, bow * 1.0, word_compact)
        mu, log_sigma = self.encode(bow)
        return self.decode(mu, log_sigma, word_compact)
