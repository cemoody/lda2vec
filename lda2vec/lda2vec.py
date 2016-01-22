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

    def __init__(self, n_words, n_hidden, counts, n_samples=5, grad_clip=5.0,
                 gpu=None, logging_level=0, dropout_ratio=0.5, window=5):
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
        dropout_ratio : float
            Ratio of elements in the context to dropout when training
        window : int
            Number of words to look behind and ahead of every context word

        >>> from lda2vec import LDA2Vec
        >>> n_words = 10
        >>> n_docs = 15
        >>> n_hidden = 8
        >>> n_topics = 2
        >>> n_obs = 300
        >>> words = np.random.randint(n_words, size=(n_obs))
        >>> _, counts = np.unique(words, return_counts=True)
        >>> model = LDA2Vec(n_words, n_hidden, counts)
        >>> model.add_categorical_feature(n_docs, n_topics, name='document id')
        >>> model.finalize()
        >>> doc_ids = np.arange(n_obs) % n_docs
        >>> loss = model.fit_partial(words, 1.0, categorical_features=doc_ids)
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging_level)
        self.logger.info("Setup LDA2Vec class")

        assert len(counts) <= n_words
        self.counts = counts
        self.frequency = counts / np.sum(counts)
        self.n_words = n_words
        self.n_hidden = n_hidden
        self.n_samples = n_samples
        self.grad_clip = grad_clip
        self.dropout_ratio = dropout_ratio
        self.window = window
        self.categorical_features = {}
        self.categorical_feature_names = []
        self.categorical_feature_counts = {}

    def add_categorical_feature(self, n_possible_values, n_latent_factors,
                                loss_type=None, n_target_out=None, name=None):
        """ Add a categorical feature to the context. You must add categorical_features
        in the order in which they'll appear when `fit` is called. Optionally
        make it a supervised feature.

        Arguments
        ---------
        n_possible_values : int
            The maximum index this feature attains. E.g., the total number of
            documents.
        n_latent_factors : int
            Each unique feature in the category wil be decomposed into this
            number of latent factors.
        loss_type : str
            String representing a chainer loss function. Must be in
            ['sigmoid_cross_entropy', 'softmax_cross_entropy',
             'hinge', 'mean_squared_error']
        """
        em = EmbedMixture(n_possible_values, n_latent_factors, self.n_hidden)
        transform, loss_func = None, None
        if name is None:
            name = "categorical_feature_%0i" % (len(self.categorical_features))
        if loss_type is not None:
            transform = L.Linear(n_latent_factors, n_target_out)
            assert loss_type in self._loss_types
            assert loss_type in dir(chainer.functions)
            loss_func = getattr(chainer.functions, loss_type)
            msg = "Added categorical feature %s with loss function %s"
            self.logger.info(msg % (name, loss_type))
        else:
            self.logger.info("Added categorical_feature %s" % name)
        self.categorical_features[name] = (em, transform, loss_func)
        counts = np.zeros(n_possible_values).astype('int32')
        self.categorical_feature_counts[name] = counts
        self.categorical_feature_names.append(name)

    def finalize(self):
        loss_func = L.NegativeSampling(self.n_hidden, self.counts,
                                       self.n_samples)
        data = np.random.randn(len(self.counts), self.n_hidden)
        data /= np.sqrt(np.prod(data.shape))
        loss_func.W.data[:] = data[:].astype('float32')
        kwargs = dict(vocab=L.EmbedID(self.n_words, self.n_hidden),
                      loss_func=loss_func)
        for name, (em, transform, lf) in self.categorical_features.items():
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

    def _context(self, data_categorical_features):
        """ For every context calculate and sum the embedding."""
        context = None
        for j, data in enumerate(data_categorical_features):
            cat_feat_name = self.categorical_feature_names[j]
            d = self[cat_feat_name + "_mixture"](data)
            e = F.dropout(d, ratio=self.dropout_ratio)
            context = e if context is None else context + e
        return context

    def _priors(self):
        """ Measure likelihood of seeing topic proportions"""
        loss = None
        for categorical_feature_name in self.categorical_feature_names:
            name = categorical_feature_name + "_mixture"
            dl = dirichlet_likelihood(self[name].weights)
            loss = dl if loss is None else dl + loss
        return loss

    def _loss(self, context, target, weight):
        _context = F.dropout(context, ratio=self.dropout_ratio)
        _word = F.dropout(self.vocab(target), ratio=self.dropout_ratio)
        dot = -F.log(F.sigmoid(F.sum(_context * _word, axis=1)) + 1e-9)
        return F.sum(dot * weight)

    def _neg_sample(self, context, target, weight):
        batchsize = target.data.shape[0]
        loss = self._loss(context, target, weight)
        samples = self.loss_func.sampler.sample((self.n_samples, batchsize))
        for sample in samples:
            sample = Variable(self.xp.asarray(sample))
            loss += self._loss(-context, sample, weight)
        return loss

    def _skipgram_flat(self, words, cat_feats, ignore_below=3):
        if type(cat_feats) is not list:
            cat_feats = [cat_feats]
        window = self.window
        xp = self.xp
        loss = None
        cwords = xp.asarray(words)
        to_var = lambda x: Variable(xp.asarray(x.astype('int32')))
        vcat_feats = [to_var(cf[window: -(window + 1)]) for cf in cat_feats]
        cntxt = self._context(vcat_feats)
        pivot = Variable(cwords[window: -(window + 1)])
        cntxt += self.vocab(pivot)
        for offset in range(-window, window + 1):
            weight = None
            for cf in cat_feats:
                w = (cf[window: -(window + 1)] ==
                     cf[window + offset: -(window + 1) + offset])
                weight = w if weight is None else np.logical_and(w, weight)
            weight = Variable(xp.asarray(weight * 1.0).astype('float32'))
            target = Variable(cwords[window + offset: -(window + 1) + offset])
            l = self._neg_sample(cntxt, target, weight)
            loss = l if loss is None else loss + l
        loss.backward()
        return loss.data + 0.0

    def _target(self, data_cat_feats, data_targets):
        losses = None
        args = (data_cat_feats, data_targets, self.categorical_feature_names)
        for data_cat_feat, data_target, cat_feat_name in zip(*args):
            cat_feat = self.categorical_features[cat_feat_name]
            embedding, transform, loss_func = cat_feat
            # This function will input an ID and ouput
            # (batchsize, n_hidden)
            latent = embedding(data_cat_feat)
            # Transform (batchsize, n_hidden) -> (batchsize, n_dim)
            # n_dim is 1 for RMSE, 1 for logistic outcomes, n for softmax
            output = transform(latent)
            # Loss_func gives likelihood of data_target given output
            l = loss_func(output, data_target)
            losses = l if losses is None else losses + l
        if losses is None:
            losses = 0.0
        return losses

    def _check_input(self, word_matrix, categorical_features, targets):
        if word_matrix is not None:
            word_matrix = word_matrix.astype('int32')
        if self._finalized is False:
            self.finalize()
        if isinstance(categorical_features, (np.ndarray, np.generic)):
            # If we pass in a single categorical feature, wrap it into a list
            categorical_features = [categorical_features]
        if isinstance(targets, (np.ndarray, np.generic)):
            # If we pass in a single target, wrap it into a list
            targets = [targets]
        if categorical_features is None:
            categorical_features = []
        else:
            msg = "Number of categorical features not equal to initialized"
            test = len(categorical_features) == len(self.categorical_features)
            assert test, msg
            to_var = lambda c: Variable(self.xp.asarray(c.astype('int32')))
            categorical_features = [to_var(c) for c in categorical_features]
        if targets is None:
            targets = []
        else:
            msg = "Number of targets not equal to initialized no. of targets"
            vals = self.categorical_features.values()
            assert len(targets) == sum([c[2] is not None for c in vals])
        for i, categorical_feature in enumerate(categorical_features):
            msg = "Number of rows in word matrix unequal"
            msg += "to that in categorical feature #%i" % i
            if word_matrix is not None:
                assert word_matrix.shape[0] == \
                    categorical_feature.data.shape[0], msg
        for i, target in enumerate(targets):
            msg = "Number of rows in word matrix unequal"
            msg += "to that in target array %i" % i
            if word_matrix is not None:
                assert word_matrix.shape[0] == target.data.shape[0], msg
        return word_matrix, categorical_features, targets

    def _log_prob_words(self, context, temperature=1.0):
        """ This calculates a softmax over the vocabulary as a function
        of the dot product of context and word.
        """
        dot = F.matmul(context, F.transpose(self.vocab.W))
        prob = F.softmax(dot / temperature)
        return F.log(prob)

    def log_prob_words(self, categorical_features, temperature=1.0):
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
        categorical_features: int array
            Array of categorical_features that compute the context
        temperature : float, default=1.0
            Inifinite temperature encourages all probability to spread out
            evenly, but zero temperature encourages the probability to be
            mapped to a single word.

        See also:
        ---------
        http://qpleple.com/perplexity-to-evaluate-topic-models/
        """
        _, categorical_features, _ = \
            self._check_input(None, categorical_features, None)
        msg = "Temperature must be non-negative"
        assert temperature > 0.0, msg
        if self._finalized is False:
            self.finalize()
        context = self._context(categorical_features)
        log_prob = self._log_prob_words(context, temperature=temperature)
        return log_prob

    def compute_log_perplexity(self, words_flat, categorical_features=None,
                               temperature=1.0):
        """ Compute the log perplexity given the categorical_features and a validation
        set of words.

        :math:`log\_perplexity=\frac{-\Sigma_d log(p(w_d))}{N}`

        We ignore the negative sampling part of the objective and focus on
        the positive categorical_feature to rpedict perplexity:
        :math:`p(w_d)=\sigma(x^\\top w_p)`

        Arguments
        ---------
        words_flat : int array
            Array of ground truth (correct) words
        categorical_features : list of int arrays
            Each categorical_feature in this array represents the context
            behind every word
        temperature : float
            High temperature spread probability over all words, low
            temperatures concentrate it on the most common word.
        """
        words_flat, categorical_features, _ = \
            self._check_input(words_flat, categorical_features, None)
        context = self._context(categorical_features)
        n_words = words_flat.shape[0]
        log_prob = self._log_prob_words(context, temperature=temperature)
        # http://qpleple.com/perplexity-to-evaluate-topic-models/
        log_perp = -F.sum(log_prob) / n_words
        return log_perp

    def _update_comp_counts(self, categorical_features):
        if not isinstance(categorical_features, list):
            categorical_features = [categorical_features, ]
        for j, categorical_feature in enumerate(categorical_features):
            name = self.categorical_feature_names[j]
            uniques, counts = np.unique(categorical_feature,
                                        return_counts=True)
            self.categorical_feature_counts[name][uniques] += counts

    def fit_partial(self, words_flat, fraction, categorical_features=None,
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
        categorical_features : int array
            List of arrays. Each array is aligned with `words_flat`. Each
            array details the categorical_feature a word is associated with,
            for example a document index or a user index.
        targets : float or int arrays
            This is usually side information related to a document. Latent
            categorical_features are chosen so that targets will correlated
            with them. For example, this could be the sold outcome of a client
            comment or the number of votes a comment receives.
        """
        self._n_partial_fits += 1
        self._update_comp_counts(categorical_features)
        words_flat, vcategorical_features, targets = \
            self._check_input(words_flat, categorical_features, targets)
        # Before calculating gradients, zero them or we will get NaN
        self.zerograds()
        prior_loss = self._priors()
        words_loss = self._skipgram_flat(words_flat, categorical_features)
        trget_loss = self._target(vcategorical_features, targets)
        # Loss is composed of loss from predicting the word given context,
        # the target given the context, and the loss due to the prior
        # on the mixture embedding
        total_loss = trget_loss + prior_loss * fraction * (self.window * 2.0)
        # Calculate back gradients
        total_loss.backward()
        # Propagate gradients
        self._optimizer.update()
        msg = "Partial fit loss: %1.5e Prior: %1.5e"
        msg = msg % (words_loss, prior_loss.data * fraction)
        self.logger.info(msg)
        return total_loss

    def fit(self, words_flat, categorical_features=None, targets=None,
            epochs=10, fraction=0.01):
        """ Train the latent document-to-topic weights, topic vectors,
        and word vectors on the full dataset.

        Arguments
        ---------
        words_flat : int array
            A flattened 1D array of shape (n_observations) where each row is
            in a single document.
        fraction : float
            Break the data into chunks of this fractional size
        categorical_features : int array
            List of arrays. Each array is aligned with `words_flat`. Each
            array details the categorical_feature a word is associated with,
            for example a document index or a user index.
        targets : float or int arrays
            This is usually side information related to a document. Latent
            categorical_features are chosen so that targets will correlated
            with them. For example, this could be the sold outcome of a
            client comment or the number of votes a comment receives.
        epochs : int
            Number of epochs to train over the whole dataset
        """
        n_chunk = int(words_flat.shape[0] * fraction)
        for epoch in range(epochs):
            args = []
            if categorical_features is not None:
                args += categorical_features
            if targets is not None:
                args += targets
            for chunk, doc_id in _chunks(n_chunk, words_flat, *args):
                this_fraction = len(chunk) * 1.0 / words_flat.shape[0]
                self.fit_partial(chunk, this_fraction,
                                 categorical_features=[doc_id])

    def prepare_topics(self, categorical_feature_name, vocab, temperature=1.0):
        """ Collects a dictionary of word, document and topic distributions.

        Arguments
        ---------
        categorical_feature_name : str or int
            If the categorical_feature was added with a name, then specify
            the name. Otherwise the index for that categorical_feature.
        vocab : list of str
            These must be the strings for words corresponding to
            indices [0, n_words]
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
        if isinstance(categorical_feature_name, str):
            featname = categorical_feature_name
        else:
            featname = self.categorical_feature_names[categorical_feature_name]
        categorical_feature = self.categorical_features[featname]
        topic_to_word = []
        for factor_vector in categorical_feature[0].factors.W.data:
            fv = Variable(self.xp.asarray(factor_vector[None, :]))
            factor_to_word = self._log_prob_words(fv, temperature=temperature)
            topic_to_word.append(np.ravel(np.exp(factor_to_word.data)))
            msg = "Vocabulary size did not match expectation"
            assert len(vocab) == factor_to_word.data.shape[1], msg
        topic_to_word = np.array(topic_to_word)
        msg = "Not all rows in topic_to_word sum to 1"
        assert np.allclose(np.sum(topic_to_word, axis=1), 1), msg
        # Collect document-to-topic distributions, e.g. theta
        doc_to_topic = F.softmax(categorical_feature[0].weights.W).data
        msg = "Not all rows in doc_to_topic sum to 1"
        assert np.allclose(np.sum(doc_to_topic, axis=1), 1), msg
        # Collect document lengths
        doc_lengths = self.categorical_feature_counts[featname].astype('int32')
        # Collect word frequency
        term_frequency = self.counts
        data = {'topic_term_dists': topic_to_word,
                'doc_topic_dists': doc_to_topic,
                'doc_lengths': doc_lengths,
                'vocab': vocab,
                'term_frequency': term_frequency}
        return data

    def top_words_per_topic(self, categorical_feature_name, vocab,
                            temperature=1.0, top_n=10):
        # Collect topic-to-word distributions, e.g. phi
        data = self.prepare_topics(categorical_feature_name, vocab,
                                   temperature=temperature)
        for j, topic_to_word in enumerate(data['topic_term_dists']):
            top = np.argsort(topic_to_word)[::-1][:top_n]
            prefix = "Top words in topic %i " % j
            print prefix + ' '.join([data['vocab'][i] for i in top])


def _chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    for i in xrange(0, len(args[0]), n):
        yield [a[i:i+n] for a in args]
