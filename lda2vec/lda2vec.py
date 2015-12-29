import numpy as np
import six
import logging

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import Variable

from embed_mixture import EmbedMixture
from dirichlet_likelihood import dirichlet_likelihood


class LDA2Vec(chainer.Chain):
    _loss_types = ['sigmoid_cross_entropy', 'softmax_cross_entropy',
                   'hinge', 'mean_squared_error']
    _initialized = False

    def __init__(self, n_words, n_sent_length, n_hidden, counts,
                 n_samples=20, grad_clip=5.0, gpu=None, logging_level=0):
        """ LDA-like model with multiple contexts and supervised labels.
        In the LDA generative model words are sampled from a topic vector.
        In this model, words are drawn from a combination of contexts not
        limited to a single source. The objective function is then similar
        to that of word2vec, where the context is changed from a single pivot
        word to have a structure imposed by the researcher. Each context
        can then also be supervised and predictive.

        Args:
            n_sent_length (int): Maximum number of words per sentence.
            n_hidden (int): Number of dimensions in a word vector.
            counts (dict): A dictionary with keys as word indices and values
                as counts for that word index.

        >>> from lda2vec import LDA2Vec
        >>> n_words = 10
        >>> n_docs = 15
        >>> n_sent_length = 5
        >>> n_hidden = 8
        >>> words = np.random.randint(n_words, size=(n_docs, n_sent_length))
        >>> _, counts = np.unique(words, return_counts=True)
        >>> model = LDA2Vec(n_words, n_sent_length, n_hidden, counts)
        >>> model.fit_partial(words, 1.0)
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging_level)
        self.logger.info("Setup LDA2Vec class")

        self.counts = counts
        self.frequency = counts / np.sum(counts)
        self.n_words = n_words
        self.n_sent_length = n_sent_length
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        if gpu >= 0:
            self._xp = cuda.cupy
            self.logger.info("Using CUPY on the GPU")
        else:
            self._xp = np
            self.logger.info("Using NumPy on the CPU")
        self.grad_clip = grad_clip
        self.components = {}
        self.component_names = []

    def add_component(self, n_documents, n_topics, loss_type=None,
                      n_target_out=None, name=None):
        """ Add a component to the context. You must add components in the
        order in which they'll appear when `fit` is called. Optionally make
        it a supervised component.

        Args:
            n_documents (int): Number of total documents.
            n_topics (int): Number of topics for this component.
            loss_type (str): String representing a chainer loss function.
                Must be in ['sigmoid_cross_entropy', 'softmax_cross_entropy',
                            'hinge', 'mean_squared_error']
        """
        em = EmbedMixture(n_documents, n_topics, self.n_hidden)
        transform, loss_func = None, None
        if name is None:
            name = "comp_%0i" % (len(self.components))
        if loss_type is not None:
            transform = L.Linear(n_topics, n_target_out)
            assert loss_type in self._loss_types
            assert loss_type in dir(chainer.functions)
            loss_func = getattr(chainer.functions, loss_type)
            self.logger.info("Added component %s with loss function %s" %
                             (name, loss_type))
        else:
            self.logger.info("Added component %s" % name)
        self.components[name] = (em, transform, loss_func)
        self.component_names.append(name)

    def initialize(self):
        loss_func = L.NegativeSampling(self.n_hidden, self.counts,
                                       self.n_samples)
        data = np.random.randn(len(self.counts), self.n_hidden)
        data /= np.sqrt(np.prod(data.shape))
        loss_func.W.data[:] = data[:].astype('float32')
        kwargs = dict(vocab=L.EmbedID(self.n_words, self.n_hidden),
                      loss_func=loss_func)
        for name, (em, transform, lf) in self.components.items():
            kwargs[name + '_mixture'] = em
            if transform is not None:
                kwargs[name + '_linear'] = transform
        super(LDA2Vec, self).__init__(**kwargs)
        self._setup()
        self._initialized = True
        self.logger.info("Finished initializing class")

    def _setup(self):
        optimizer = optimizers.Adam()
        optimizer.setup(self)
        clip = chainer.optimizer.GradientClipping(self.grad_clip)
        optimizer.add_hook(clip)
        self._optimizer = optimizer
        self.logger.info("Setup optimizer")

    def _context(self, components):
        """ For every context calculate and sum the embedding."""
        context = None
        for component_name, component in zip(self.component_names, components):
            e = self[component_name + "_mixture"](component)
            context = e if context is None else context + e
        return context

    def _priors(self, contexts):
        """ Measure likelihood of seeing topic proportions"""
        loss = None
        for component in self.component_names:
            name = component + "_mixture"
            dl = dirichlet_likelihood(self[name].weights)
            loss = dl if loss is None else dl + loss
        return loss

    def _unigram(self, context, words, **kwargs):
        """ Given context, predict words."""
        total_loss = None
        for column in six.moves.range(self.n_sent_length):
            target = Variable(self._xp.asarray(words[:, column]))
            loss = self.loss_func(context, target, **kwargs)
            total_loss = loss if total_loss is None else total_loss + loss
        return total_loss

    def _skipgram(self, context, words):
        """ Given context + every word, predict every other word"""
        raise NotImplemented

    def _target(self, data_components, data_targets):
        losses = None
        args = (data_components, data_targets, self.component_names)
        for data_component, data_target, component in zip(*args):
            # This function will input an ID and ouput
            # (batchsize, n_hidden)
            embedding = component[0]
            # Transform (batchsize, n_hidden) -> (batchsize, n_dim)
            # n_dim is 1 for RMSE, 1 for logistic outcomes, n for softmax
            transform = component[1]
            # loss_func gives likelihood of data_target given output
            loss_func = component[2]
            latent = embedding(data_component)
            output = transform(latent)
            l = loss_func(output, data_target)
            losses = l if losses is None else losses + l
        if losses is None:
            losses = 0.0
        return losses

    def _check_input(self, word_matrix, components, targets):
        word_matrix = word_matrix.astype('int32')
        if self._initialized is False:
            self.initialize()
        if components is None:
            components = []
        else:
            msg = "Number of components not equal to initialized components"
            assert len(components) == len(self.components), msg
            components = [Variable(self._xp.asarray(c.astype('int32')))
                          for c in components]
        if targets is None:
            targets = []
        else:
            msg = "Number of targets not equal to initialized no. of targets"
            vals = self.components.values()
            assert len(targets) == sum([c[2] is not None for c in vals])
        return word_matrix, components, targets

    def compute_log_perplexity(self, word_matrix, components=None):
        """ Compute the log perplexity given the components and a validation
        set of words.

        :math:`log\_perplexity=\frac{-\Sigma_d log(p(w_d))}{N}`

        We ignore the negative sampling part of the objective and focus on
        the positive component to rpedict perplexity:
        :math:`p(w_d)=\sigma(x^\\top w_p)`
        """
        word_matrix, components, targets = self._check_input(word_matrix,
                                                             components,
                                                             None)
        context = self._context(components)
        n_words = np.prod(word_matrix.shape)
        prob = F.softmax(F.matmul(context, F.transpose(self.loss_func.W)))
        # http://qpleple.com/perplexity-to-evaluate-topic-models/
        log_perp = -F.sum(F.log(prob)) / n_words
        return log_perp

    def fit_partial(self, word_matrix, fraction, components=None,
                    targets=None):
        """ Train the latent document-to-topic weights, topic vectors,
        and word vectors on partial subset of the full data.

        Args:
            word_matrix (numpy int array): Matrix of shape
                (n_sentences, n_sent_length) where each row is a single
                sentence, nth column is the nth word in that sentence.
            fraction (float): Fraction of all words this subset represents.
        """
        word_matrix, components, targets = self._check_input(word_matrix,
                                                             components,
                                                             targets)
        context = self._context(components)
        # word_matrix_pruned = self._prune_rare(word_matrix.astype('int32'))
        prior_loss = self._priors(context)
        words_loss = self._unigram(context, word_matrix)
        trget_loss = self._target(components, targets)
        # Loss is composed of loss from predicting the word given context,
        # the target given the context, and the loss due to the prior
        # on the mixture embedding
        total_loss = words_loss + trget_loss + prior_loss * fraction
        # Before calculating gradients, zero them or we will get NaN
        self.zerograds()
        # Calculate back gradients
        total_loss.backward()
        # Propagate gradients
        self._optimizer.update()
        self.logger.info("Partial fit loss: %1.1e" % total_loss.data)
        return total_loss

    def term_topics(self, component):
        data = {'topic_term_dists': None,  # phi, [n_topics, n_words]
                'doc_topic_dists': None,  # theta, [n_docs, n_topics]
                'doc_lengths': None,
                'vocab': None,
                'term_frequency': None
                }
        return data
