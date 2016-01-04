import numpy as np
import logging

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import Variable

from embed_mixture import EmbedMixture
from dirichlet_likelihood import dirichlet_likelihood


class LDA2Vec(chainer.Chain):
    _loss_types = ['sigmoid_cross_entropy', 'softmax_cross_entropy',
                   'hinge', 'mean_squared_error']
    _finalized = False
    _n_partial_fits = 0

    def __init__(self, n_words, n_hidden, counts, n_samples=20, grad_clip=5.0,
                 gpu=None, logging_level=0):
        """ LDA-like model with multiple contexts and supervised labels.
        In the LDA generative model words are sampled from a topic vector.
        In this model, words are drawn from a combination of contexts not
        limited to a single source. The objective function is then similar
        to that of word2vec, where the context is changed from a single pivot
        word to have a structure imposed by the researcher. Each context
        can then also be supervised and predictive.

        Arguments
        ---------
        n_words : int
            Number of unique words in the vocabulary.
        n_hidden : int
            Number of dimensions in a word vector.
        counts : dict
            A dictionary with keys as word indices and values
            as counts for that word index.

        >>> from lda2vec import LDA2Vec
        >>> n_words = 10
        >>> n_docs = 15
        >>> n_hidden = 8
        >>> n_topics = 2
        >>> n_obs = 300
        >>> words = np.random.randint(n_words, size=(n_obs))
        >>> _, counts = np.unique(words, return_counts=True)
        >>> model = LDA2Vec(n_words, n_hidden, counts)
        >>> model.add_component(n_docs, n_topics, name='document id')
        >>> model.finalize()
        >>> doc_ids = np.arange(n_obs) % n_docs
        >>> loss = model.fit_partial(words, 1.0, components=doc_ids)
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging_level)
        self.logger.info("Setup LDA2Vec class")

        self.counts = counts
        self.frequency = counts / np.sum(counts)
        self.n_words = n_words
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.grad_clip = grad_clip
        self.components = {}
        self.component_names = []
        self.component_counts = {}

    def add_component(self, n_documents, n_topics, loss_type=None,
                      n_target_out=None, name=None):
        """ Add a component to the context. You must add components in the
        order in which they'll appear when `fit` is called. Optionally make
        it a supervised component.

        Arguments
        ---------
        n_documents : int
            Number of total documents.
        n_topics : int
            Number of topics for this component.
        loss_type : str
            String representing a chainer loss function. Must be in
            ['sigmoid_cross_entropy', 'softmax_cross_entropy',
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

    def finalize(self):
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
        self._finalized = True
        self.logger.info("Finalized the class")

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

    def _unigram(self, context, words_flat, window=10, **kwargs):
        """ Given context, predict words."""
        predict_word = Variable(self.xp.asarray(words_flat))
        loss = self.loss_func(context, predict_word, **kwargs)
        return loss

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
        if word_matrix is not None:
            word_matrix = word_matrix.astype('int32')
        if self._finalized is False:
            self.finalize()
        if isinstance(components, (np.ndarray, np.generic)):
            # If we pass in a single component, wrap it into a list
            components = [components]
        if isinstance(targets, (np.ndarray, np.generic)):
            # If we pass in a single target, wrap it into a list
            targets = [targets]
        if components is None:
            components = []
        else:
            msg = "Number of components not equal to initialized components"
            assert len(components) == len(self.components), msg
            components = [Variable(self.xp.asarray(c.astype('int32')))
                          for c in components]
        if targets is None:
            targets = []
        else:
            msg = "Number of targets not equal to initialized no. of targets"
            vals = self.components.values()
            assert len(targets) == sum([c[2] is not None for c in vals])
        for i, component in enumerate(components):
            msg = "Number of rows in word matrix unequal"
            msg += "to that in component array %i" % i
            if word_matrix is not None:
                assert word_matrix.shape[0] == component.data.shape[0], msg
        for i, target in enumerate(targets):
            msg = "Number of rows in word matrix unequal"
            msg += "to that in target array %i" % i
            if word_matrix is not None:
                assert word_matrix.shape[0] == target.data.shape[0], msg
        return word_matrix, components, targets

    def _log_prob_words(self, context, temperature=1.0):
        """ This calculates an softmax over the vocabulary as a function
        of the dot product of context and word.
        """
        dot = F.matmul(context, F.transpose(self.loss_func.W))
        prob = F.softmax(dot / temperature)
        return F.log(prob)

    def log_prob_words(self, component, temperature=1.0):
        """ Compute the probability of each word given context vectors.
        With negative sampling a full softmax distribution is not calculated
        and so an approximation is picking words by similarity to the context.
        The probabillity of picking a word increases with the dot product of
        the word and context. The `temperature` modulates what a 'high'
        similarity is. Temperatures large compared to the dot product
        will supress differences, and thus the probability will be spread out
        all words. At low temperatures the differences are exagerated, and
        as temperature approaches zero a single word will take the full
        probability.

        .. :math:`p\_word = softmax(context \cdot word / temperature)

        Arguments
        ---------
        component : int array
            Array of components that compute the context
        temperature : float, default=1.0
            Inifinite temperature encourages all probability to spread out
            evenly, but zero temperature encourages the probability to be
            mapped to a single word.

        See also:
        ---------
        http://qpleple.com/perplexity-to-evaluate-topic-models/
        """
        _, component, _ = self._check_input(None, component, None)
        msg = "Temperature must be non-negative"
        assert temperature > 0.0, msg
        if self._finalized is False:
            self.finalize()
        context = self._context(component)
        log_prob = self._log_prob_words(context, temperature=temperature)
        return log_prob

    def compute_log_perplexity(self, words_flat, components=None,
                               temperature=1.0):
        """ Compute the log perplexity given the components and a validation
        set of words.

        :math:`log\_perplexity=\frac{-\Sigma_d log(p(w_d))}{N}`

        We ignore the negative sampling part of the objective and focus on
        the positive component to rpedict perplexity:
        :math:`p(w_d)=\sigma(x^\\top w_p)`

        Arguments
        ---------
        words_flat : int array
            Array of ground truth (correct) words
        components : list of int arrays
            Each component in this array represents the context behind
            every word
        temperature : float
            High temperature spread probability over all words, low
            temperatures concentrate it on the most common word.
        """
        words_flat, components, _ = self._check_input(words_flat,
                                                      components,
                                                      None)
        context = self._context(components)
        n_words = words_flat.shape[0]
        log_prob = self._log_prob_words(context, temperature=temperature)
        # http://qpleple.com/perplexity-to-evaluate-topic-models/
        log_perp = -F.sum(log_prob) / n_words
        return log_perp

    def _update_comp_counts(self, components):
        for component in components:
            uniques, counts = np.unique(component, return_counts=True)
            for u, c in zip(uniques, counts):
                self.component_counts[component][u] += c

    def fit_partial(self, words_flat, fraction, components=None,
                    targets=None):
        """ Train the latent document-to-topic weights, topic vectors,
        and word vectors on partial subset of the full data.

        Arguments
        ---------
        words_flat : int array
            A flattened 1D array of shape (n_observations) where each row is
            in a single document.
        fraction : float
            Fraction of all words this subset represents. If thi
        components : int array
            List of arrays. Each array is aligned with `words_flat`. Each
            array details the component a word is associated with, for example
            a document index or a user index.
        targets : float or int arrays
            This is usually side information related to a document. Latent
            components are chosen so that targets will correlated with them.
            For example, this could be the sold outcome of a client comment
            or the number of votes a comment receives.
        """
        self._n_partial_fits += 1
        self.logger.info("Computing partial fit #%i" % self._n_partial_fits)
        words_flat, components, targets = self._check_input(words_flat,
                                                            components,
                                                            targets)
        self._update_comp_counts(components)
        context = self._context(components)
        prior_loss = self._priors(context)
        words_loss = self._unigram(context, words_flat)
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

    def fit(self, words_flat, fraction=0.01, components=None, targets=None,
            epochs=10):
        """ Train the latent document-to-topic weights, topic vectors,
        and word vectors on the full dataset.

        Arguments
        ---------
        words_flat : int array
            A flattened 1D array of shape (n_observations) where each row is
            in a single document.
        fraction : float
            Break the data into chunks of this fractional size
        components : int array
            List of arrays. Each array is aligned with `words_flat`. Each
            array details the component a word is associated with, for example
            a document index or a user index.
        targets : float or int arrays
            This is usually side information related to a document. Latent
            components are chosen so that targets will correlated with them.
            For example, this could be the sold outcome of a client comment
            or the number of votes a comment receives.
        epochs : int
            Number of epochs to train over the whole dataset
        """
        n_chunk = int(words_flat.shape[0] * fraction)
        for epoch in range(epochs):
            args = components + targets
            for chunk, doc_id in _chunks(n_chunk, words_flat, *args):
                self.fit_partial(chunk, fraction, components=[doc_id])

    def prepare_topics(self, component_name, index_to_word, temperature=1.0):
        """ Collects a dictionary of word, document and topic distributions.

        Arguments
        ---------
        component_name : str or int
            If the component was added with a name, then specify the name.
            Otherwise the index for that component.
        index_to_word : dict
            Keys must be integers and values the string representation for
            that word
        temperature : float
            Used to calculate the log probability of a word. See log_prob_words
            for a description.

        Returns
        -------
        data : dict
            This dictionary is readily consumed by pyLDAVis for topic
            visualization.
        """
        # Collect topic-to-word distributions, e.g. phi
        if type(component_name) is str:
            components = self.components[component_name]
        else:
            component_name = self.component_names[component_name]
            components = self.components[component_name]
        topic_to_word = []
        for factor_vector in components.factors.W:
            fv = Variable(self.xp.asarray(factor_vector))
            factor_to_word = self._log_prob_words(fv, temperature=temperature)
            topic_to_word.append(factor_to_word)
            assert len(index_to_word) == factor_to_word.shape[1]
        # Collect document-to-topic distributions, e.g. theta
        doc_to_topic = components.weights.W
        # Collect document lengths
        doc_lengths = None
        # Collect vocabulary list
        vocab = index_to_word
        # Collect word frequency
        term_frequency = self.counts
        data = {'topic_term_dists': topic_to_word,
                'doc_topic_dists': doc_to_topic,
                'doc_lengths': doc_lengths,
                'vocab': vocab,
                'term_frequency': term_frequency}
        return data


def _chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    for i in xrange(0, len(args[0]), n):
        yield [a[i:i+n] for a in args]
