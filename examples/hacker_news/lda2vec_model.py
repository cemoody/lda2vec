from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood
from lda2vec.utils import move

from chainer import Chain
import chainer.links as L
import chainer.functions as F
import chainer


class LDA2Vec(Chain):
    def __init__(self, n_stories, n_story_topics, n_authors, n_author_topics,
                 n_vocab, n_units=256, dropout_ratio=0.5, train=True,
                 counts=None, n_samples=15):
        sm = EmbedMixture(n_stories, n_story_topics, n_units,
                          dropout_ratio=dropout_ratio)
        am = EmbedMixture(n_authors, n_author_topics, n_units,
                          dropout_ratio=dropout_ratio)
        kwargs = {}
        kwargs['mixture_stories'] = sm
        kwargs['mixture_authors'] = am
        kwargs['embed'] = L.EmbedID(n_vocab, n_units)
        kwargs['sampler'] = L.NegativeSampling(n_units, counts, n_samples)
        super(LDA2Vec, self).__init__(**kwargs)
        self.n_units = n_units
        self.train = train
        self.dropout_ratio = dropout_ratio
        self.n_samples = n_samples

    def prior(self):
        dl1 = dirichlet_likelihood(self.mixture_stories.weights)
        dl2 = dirichlet_likelihood(self.mixture_authors.weights)
        return dl1 + dl2

    def loss(self, source, target, weight):
        word = F.dropout(self.embed(target), ratio=self.dropout_ratio)
        inner = F.sum(source * word, axis=1)
        sp = F.sum(F.softplus(-inner) * weight)
        return sp

    def fit_partial(self, rsty_ids, raut_ids, rwrd_ids, window=5):
        doc_idx, usr_idx, wrd_idx = move(self.xp, rsty_ids, raut_ids, rwrd_ids)
        pivot = self.embed(next(move(self.xp, rwrd_ids[window: -window])))
        sty_at_pivot = rsty_ids[window: -window]
        aut_at_pivot = raut_ids[window: -window]
        sty = self.mixture(next(move(self.xp, sty_at_pivot)))
        aut = self.mixture(next(move(self.xp, aut_at_pivot)))
        loss = 0.0
        start, end = window, rwrd_ids.shape[0] - window
        context = (F.dropout(sty, self.dropout_ratio) +
                   F.dropout(aut, self.dropout_ratio) +
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
            # Predict word given context and pivot word
            # The target starts before the pivot
            # Skip predicting the current pivot
            if frame == 0:
                continue
            # Here we're creating a weight mask. We don't want to
            # predict tokens that are outside this document or user
            # scope.
            wrd_at_target = rwrd_ids[start + frame: end + frame]
            sty_at_target = rsty_ids[start + frame: end + frame]
            aut_at_target = raut_ids[start + frame: end + frame]
            doc_is_same = sty_at_target == sty_at_pivot
            usr_is_same = aut_at_target == aut_at_pivot
            is_same = doc_is_same & usr_is_same
            weight, = move(self.xp, is_same.astype('float32'))
            target, = move(self.xp, wrd_at_target)
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
