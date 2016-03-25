from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

from chainer import Chain
import chainer.links as L
import chainer.functions as F
import chainer


class LDA2Vec(Chain):
    def __init__(self, n_documents=100, n_document_topics=10,
                 n_units=256, n_vocab=1000, dropout_ratio=0.5, train=True,
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

    def loss(self, source, target, weight):
        word = F.dropout(self.embed(target), ratio=self.dropout_ratio)
        dot = -F.log(F.sigmoid(F.sum(source * word, axis=1)) + 1e-9)
        return F.sum(dot * weight)

    def fit_partial(self, rdoc_ids, rword_indices, window=5):
        doc_ids, word_indices = move(self.xp, rdoc_ids, rword_indices)
        pivot = self.embed(next(move(self.xp, rword_indices[window: -window])))
        doc_at_pivot = rdoc_ids[window: -window]
        doc = self.mixture(next(move(self.xp, doc_at_pivot)))
        loss = 0.0
        start, end = window, rword_indices.shape[0] - window
        context = (F.dropout(doc, self.dropout_ratio) +
                   F.dropout(pivot, self.dropout_ratio))
        n_frame = 2 * window
        # Precompute all neg samples since they're indep of frame
        size = context.data.shape[0]
        samples = self.sampler.sampler.sample((self.n_samples * n_frame, size))
        samples = chainer.cuda.cupy.split(samples.ravel(), n_frame)
        sources = []
        targets = []
        weights = []
        for frame in range(-window, window + 1):
            # Skip predicting the current pivot
            if frame == 0:
                continue
            # Predict word given context and pivot word
            # The target starts before the pivot
            targetidx = rword_indices[start + frame: end + frame]
            doc_at_target = rdoc_ids[start + frame: end + frame]
            doc_is_same = doc_at_target == doc_at_pivot
            weight, = move(self.xp, doc_is_same.astype('float32'))
            target, = move(self.xp, targetidx)
            sources.append(context)
            targets.append(target)
            weights.append(weight)
            sample, = move(self.xp, samples.pop())
            targets.append(sample)
            for _ in range(self.n_samples):
                # Note that the context is now negative
                sources.append(-context)
                weights.append(weight)
        sources = F.concat(sources, axis=0)
        targets = F.concat(targets, axis=0)
        weights = F.concat(weights, axis=0)
        loss = self.loss(sources, targets, weights)
        return loss
