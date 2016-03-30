import chainer
import chainer.links as L
import chainer.functions as F

from lda2vec import utils


class NVDM(chainer.Chain):
    def __init__(self, n_vocab, n_dim):
        super(NVDM, self).__init__(l1=L.Linear(n_vocab, n_dim),
                                   l2=L.Linear(n_dim, n_dim),
                                   mu_logsigma=L.Linear(n_dim, n_dim * 2),
                                   embedding=L.Linear(n_dim, n_vocab))
        self.n_vocab = n_vocab
        self.n_dim = n_dim

    def encode(self, bow):
        """ Convert the bag of words vector of shape (n_docs, n_vocab)
        into latent mean log variance vectors.
        """
        lam = F.relu(self.l1(bow))
        pi = F.relu(self.l2(lam))
        mu, log_sigma = F.split_axis(self.mu_logsigma(pi), 2, 1)
        sample = F.gaussian(mu, log_sigma)
        loss = F.gaussian_kl_divergence(mu, log_sigma)
        return sample, loss

    def decode(self, sample, bow):
        """ Decode latent document vectors back into word counts
        (n_docs, n_vocab).
        """
        logprob = F.log_softmax(self.embedding(sample))
        # This is equivalent to a softmax_cross_entropy where instead of
        # guessing 1 of N words we have repeated observations
        # Normal softmax for guessing the next word is:
        # t log softmax(x), where t is 0 or 1
        # Softmax for guessing word counts is simply doing
        # the above more times, so multiply by the count
        # count log softmax(x)
        loss = -F.sum(bow * logprob)
        return loss

    def observe(self, bow):
        bow, = utils.move(self.xp, bow * 1.0)
        sample, kl = self.encode(bow)
        rec = self.decode(sample, bow)
        return rec, kl
