import numpy as np
import six

import chainer
import chainer.links as L

from chainer import cuda
from chainer import optimizers
from chainer import Variable

from lda2vec import EmbedMixture
from lda2vec import dirichlet_likelihood


class lda2vec(chainer.Chain):
    component_names = []
    _loss_types = ['sigmoid_cross_entropy', 'softmax_cross_entropy',
                   'hinge', 'mean_squared_error']

    def __init__(self, n_words, n_sent_length, n_hidden, counts,
                 n_samples=20, grad_clip=5.0, gpu=0):
        """ LDA-like model with multiple contexts and supervised labels.
        In the LDA generative model words are sampled from a topic vector.
        In this model, words are drawn from a combination of contexts not
        limited to a single source. Each context can also be supervised
        and predictive.

        Args:
            n_sent_length (int): Maximum number of words per sentence.
            n_hidden (int): Number of dimensions in a word vector.
            counts (dict): A dictionary with keys as word indices and values
                as counts for that word index.
        """
        self.n_words = n_words
        self.n_sent_length = n_sent_length
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.xp = cuda.cupy if gpu >= 0 else np
        self.loss_func = L.NegativeSampling(n_hidden, counts, n_samples)
        self.grad_clip = grad_clip

    def add_component(self, n_documents, n_topics, loss_type=None,
                      n_target_out=None, name=None):
        """ Add a component to the context. Optionally make it a
        supervised component.

        Args:
            n_documents (int): Number of total documents.
            n_topics (int): Number of topics for this component.
            loss_type (str): String representing a chainer loss function.
                Must be in ['sigmoid_cross_entropy', 'softmax_cross_entropy',
                            'hinge', 'mean_squared_error']
        """
        em = EmbedMixture(n_documents, n_topics, self.n_hidden)
        transform, loss_func = None, None
        if loss_type is not None:
            transform = L.Linear(n_topics, n_target_out)
            assert loss_type in self._loss_types
            loss_func = getattr(chainer.functions, loss_type)
        self.components[name] = (em, transform, loss_func)

    def initialize(self):
        kwargs = dict(vocab=L.EmbedID(self.n_vocab, self.n_hidden))
        for name, (em, transform, lf) in zip(self.components.items()):
            kwargs[name + '_mixture'] = em
            kwargs[name + '_linear'] = transform
            self.component_names.append(name + '_mixture')
        super(lda2vec, self).__init__(**kwargs)
        self._setup()

    def _setup(self):
        optimizer = optimizers.Adam()
        optimizer.setup(self)
        clip = chainer.optimizer.GradientClipping(self.grad_clip)
        optimizer.add_hook(clip)
        self._optimizer = optimizer

    def _context(self, contexts):
        """ For every context calculate and sum the embedding."""
        context = None
        for component_name, context in zip(self.component_names, contexts):
            e = self[component_name](context)
            context = e if context is None else context + e
        return context

    def _priors(self, contexts):
        """ Measure likelihood of seeing topic proportions"""
        loss = None
        for component in self.component_names:
            dl = dirichlet_likelihood(self[component].weights)
            loss = dl if loss is None else dl + loss
        return loss

    def _unigram(self, context, words):
        """ Given context, predict words."""
        total_loss = None
        for column in six.moves.range(self.n_sent_length):
            target = Variable(self.xp.asarray(words[:, column]))
            loss = self.loss_func(context, target)
            total_loss = loss if total_loss is None else total_loss + loss
        return total_loss

    def _skipgram(self, context, words):
        """ Given context + every word, predict every other word"""
        raise NotImplemented

    def _target(self, data_contexts, data_targets):
        tk = self.target.keys()
        args = (data_contexts, tk, data_targets)
        losses = None
        for component, context, func, target in zip(*args):
            l = func(context, target)
            losses = l if losses is None else losses + l
        return losses

    def fit_partial(self, word_matrix, fraction, contexts=None, targets=None):
        """ Train the latent document-to-topic weights, topic vectors,
        and word vectors on partial subset of the full data.

        Args:
            word_matrix (numpy int array): Matrix of shape
                (n_sentences, n_sent_length) where each row is a single
                sentence, nth column is the nth word in that sentence.
            fraction (float): Fraction of all words this subset represents.
        """
        if contexts is None:
            contexts = []
        if targets is None:
            targets = []
        msg = "Number of contexts not equal to initialized number of contexts"
        assert len(contexts) == len(self.contexts), msg
        msg = "Number of targets not equal to initialized number of targets"
        assert len(targets) == len(self.targets)
        contexts = [Variable(c.astype('int32')) for c in contexts]
        context = self._context(contexts)
        prior_loss = self._priors(contexts)
        words_loss = self._unigram(context, word_matrix.astype('int32'))
        trget_loss = self._target(contexts, targets)
        total_loss = prior_loss * fraction + words_loss + trget_loss
        self.zerograds()
        total_loss.backward()
        self._optimizer.update()
