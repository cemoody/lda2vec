from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

from chainer import Chain
import chainer.links as L
import chainer.functions as F


class LDA2Vec(Chain):
    def __init__(self, n_documents=100, n_document_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.0, train=True,
                 counts=None, n_samples=15):
        em = EmbedMixture(n_documents, n_document_topics, n_units,
                          dropout_ratio=dropout_ratio)
        kwargs = {}
        kwargs['mixture'] = em
        kwargs['embed'] = L.EmbedID(n_vocab, n_units)
        kwargs['sampler'] = L.NegativeSampling(n_units, counts, n_samples)
        super(LDA2Vec, self).__init__(**kwargs)
        self.n_units = n_units
        self.train = train
        self.dropout_ratio = dropout_ratio
        self.n_samples = n_samples

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture.weights)
        return dl1

    def fit_partial(self, rdoc_ids, rword_indices, window=5):
        n_frames = 2 * window
        doc_ids, word_indices = move(self.xp, rdoc_ids, rword_indices)
        pivot = self.embed(next(move(self.xp, rword_indices[window: -window])))
        context_at_pivot = rdoc_ids[window: -window]
        context = self.mixture(next(move(self.xp, context_at_pivot)))
        loss = 0.0
        start, end = window, rword_indices.shape[0] - window
        combined = (F.dropout(context, self.dropout_ratio) +
                    F.dropout(pivot, self.dropout_ratio))
        combineds = F.concat((combined, ) * n_frames, axis=0)
        targets = []
        for frame in range(-window, window + 1):
            # Skip predicting the current pivot
            if frame == 0:
                continue
            # Predict word given context and pivot word
            # The target starts before the pivot
            targetidx = rword_indices[start + frame: end + frame]
            context_at_target = rdoc_ids[start + frame: end + frame]
            context_same = context_at_target == context_at_pivot
            targetidx[~context_same] = -1
            target, = move(self.xp, targetidx)
            targets.append(target)
        targets = F.concat(targets, axis=0)
        loss = self.sampler(combineds, targets)
        return loss
