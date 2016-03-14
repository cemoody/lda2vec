import numpy as np
import logging
import random
import time

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import Variable

from embed_mixture import EmbedMixture
from dirichlet_likelihood import dirichlet_likelihood


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '%2.4f sec %r' % (te - ts, method.__name__)
        return result
    return timed


class LDA2Vec(chainer.Chain):
    _loss_types = ['sigmoid_cross_entropy', 'softmax_cross_entropy',
                   'hinge', 'mean_squared_error']
    _finalized = False
    _n_partial_fits = 0

    def __init__(self, n_words, n_hidden, counts, n_samples=5, grad_clip=None,
                 logging_level=0, dropout_ratio=0.2, dropout_word=0.3,
                 window=5):
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
        dropout_word : float
            With given probability, remove word from training set.
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
        self.dropout_word = dropout_word
        self.dropout_ratio = dropout_ratio
        self.window = window
        self.categorical_features = {}
        self.categorical_feature_names = []
        self.categorical_feature_counts = {}
        self.target_losses = {}

    def add_categorical_feature(self, n_possible_values, n_latent_factors,
                                covariance_penalty=None, loss_type=None,
                                n_target_out=1, name=None, l2_penalty=None,
                                logdet_penalty=None,
                                use_predefined_feature=None):
        """ Add a categorical feature to the context. You must add
        categorical_features in the order in which they'll appear when `fit`
        is called. Optionally make it a supervised feature. If you use the
        name of a feature that has already been added, the topics will be
        shared with that feature.

        Arguments
        ---------
        n_possible_values : int
            The maximum index this feature attains. E.g., the total number of
            documents.
        n_latent_factors : int
            Each unique feature in the category wil be decomposed into this
            number of latent factors.
        covariance_penalty : None, float
            If None, do not penalize covariance among topics in this feature.
            If float, larger covariance parameters discourage correlated topics
            and encourage more independent topics.
        loss_type : str
            String representing a chainer loss function. Must be in
            ['sigmoid_cross_entropy', 'softmax_cross_entropy',
             'hinge', 'mean_squared_error']
        n_target_out : int
            Dimensionality of the target. If predicting a scalar output, this
            should be 1. Otherwise should be the rank of the output.
        l2_penalty : float
            If the categorical feature is spatially continuous, like week
            number or latitude, this enforces that nearby points are similar.
        logdet_penalty : None, float
            Penalize the log determinant of the topic covariance matrix. This
            penalizes topics that are highly correlated.
        use_predefined_feature: None, str
            Instead of generating a new topic space for this feature use
            another feature's topics. This is helpful when you want the
            document topics to be exactly the same as the temporal or user
            topics.
        """
        if use_predefined_feature is not None:
            em = self.categorical_features[use_predefined_feature][0]
        else:
            em = EmbedMixture(n_possible_values, n_latent_factors,
                              self.n_hidden, dropout_ratio=self.dropout_ratio)
        transform, loss_func = None, None
        if name is None:
            name = "categorical_feature_%0i" % (len(self.categorical_features))
        if loss_type is not None:
            assert loss_type in self._loss_types
            assert loss_type in dir(chainer.functions)
            assert n_target_out is not None
            assert n_latent_factors is not None
            transform = L.Linear(n_latent_factors, n_target_out)
            loss_func = getattr(chainer.functions, loss_type)
            msg = "Added categorical feature %s with loss function %s"
            self.logger.info(msg % (name, loss_type))
        else:
            self.logger.info("Added categorical_feature %s" % name)
        self.categorical_features[name] = (em, transform, loss_func,
                                           covariance_penalty, l2_penalty,
                                           logdet_penalty)
        counts = np.zeros(n_possible_values).astype('int32')
        self.categorical_feature_counts[name] = counts
        self.categorical_feature_names.append(name)

    def add_target(self, name, target_dtype, out_size, loss_type):
        """ Add a target variable predicted using all categorical features.
        As compared to `add_categorical_features`, this feature will correlate
        all topics with the outcome instead of a single feature with a single
        outcome.

        Arguments
        ---------
        name : str
            Name of the target variable to predict. When target data
            passed to the `fit` method must match this name.
        target_dtype : 'int32' or 'float32'
            You must state if you are predicting a categorical, numerical
        out_size : int
            If the outcome is a scalar this should be 1, if the outcome is
            categorical this should be the number of categories, if the
            outcome is a vector, this should be the dimensionality of that
            vector.
        loss_type : str name of a function in chainer.loss_functions
            Must be a loss function recognized in Chainer.
        """
        msg = "target_dtype must be int32 or float32"
        assert target_dtype in ('int32', 'float32'), msg
        n_features = sum(cf[0].in_size for cf in self.categorical_features)
        transform = L.Linear(n_features, out_size)
        assert loss_type in self._loss_types
        assert loss_type in dir(chainer.functions)
        loss_func = getattr(chainer.functions, loss_type)
        msg = "Added loss function %s"
        self.logger.info(msg % loss_type)
        self.target_losses[name] = (transform, loss_func, target_dtype)

    def finalize(self):
        loss_func = L.NegativeSampling(self.n_hidden, self.counts,
                                       self.n_samples)
        data = np.random.randn(len(self.counts), self.n_hidden)
        data /= np.sqrt(np.prod(data.shape))
        loss_func.W.data[:] = data[:].astype('float32')
        kwargs = dict(vocab=L.EmbedID(self.n_words, self.n_hidden),
                      loss_func=loss_func)
        added = set()
        for name, (em, transform, lf, cp, lp, ldp) in \
                self.categorical_features.items():
            if name not in added:
                kwargs[name + '_mixture'] = em
                added.add(name)
            if transform is not None:
                kwargs[name + '_linear'] = transform
        super(LDA2Vec, self).__init__(**kwargs)
        self._setup()
        self._finalized = True
        self.logger.info("Finalized the class")

    def _setup(self):
        optimizer = optimizers.Adam()
        optimizer.setup(self)
        if self.grad_clip is not None:
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
        for cat_feat_name, vals in self.categorical_features.items():
            (embedding, transform, loss_func, penalty, l2_penalty,
                logdet_penalty) = vals
            name = cat_feat_name + "_mixture"
            dl = dirichlet_likelihood(self[name].weights)
            if penalty:
                factors = self[name].factors.W
                cc = F.cross_covariance(factors, factors)
                dl += cc
            if logdet_penalty:
                factors = self[name].factors.W
                mean = F.sum(factors, axis=0) / factors.data.shape[0]
                factors, mean = F.broadcast(factors, mean)
                factors = factors - mean
                eye = self.xp.eye(factors.data.shape[0], dtype='float32')
                cov = F.matmul(factors, F.transpose(factors)) + Variable(eye)
                ld = F.log(F.det(cov))
                dl += ld * logdet_penalty
            if l2_penalty:
                weights = self[name].weights.W
                top = Variable(weights.data[None, 0, :].T)
                bot = Variable(weights.data[None, -1, :].T)
                left = F.transpose(F.concat((top, F.transpose(weights))))
                rght = F.transpose(F.concat((F.transpose(weights), bot)))
                lp = F.mean_squared_error(left, rght)
                dl += lp
            loss = dl if loss is None else dl + loss
        return loss

    def _neg_sample(self, context, target, weight):
        batchsize = target.shape[0]
        pos = target[None, :]
        neg = self.loss_func.sampler.sample((self.n_samples, batchsize))
        samples = Variable(self.xp.concatenate((pos, neg)))
        words = F.dropout(self.vocab(samples), ratio=self.dropout_ratio)
        one = self.xp.ones_like(pos)
        zero = self.xp.zeros_like(neg)
        targets = Variable(self.xp.concatenate((one, zero)))
        # words is shape (n_samples, batchsize, dim)
        # context is shape (batchsize, dim)
        bcontext, bwords = F.broadcast(context, words)
        inner = F.sum(bcontext * bwords, axis=2)
        loss = F.sigmoid_cross_entropy(inner, targets)
        return loss

    def _skipgram_flat(self, words, cat_feats, ignore_below=3):
        if type(cat_feats) is not list:
            cat_feats = [cat_feats]
        window = self.window
        xp = self.xp
        loss = None
        cwords = xp.asarray(words)
        vcat_feats = [self.to_var(cf[window: -(window + 1)])
                      for cf in cat_feats]
        cntxt = self._context(vcat_feats)
        pivot = Variable(cwords[window: -(window + 1)])
        cntxt += self.vocab(pivot)
        for offset in range(-window, window + 1):
            weight = None
            for cf in cat_feats:
                w = (cf[window: -(window + 1)] ==
                     cf[window + offset: -(window + 1) + offset])
                weight = w if weight is None else np.logical_and(w, weight)
            if self.dropout_word:
                wd = np.random.uniform(0, 1, weight.shape[0])
                wd = (wd > self.dropout_word).astype('bool')
                weight = np.logical_and(weight, wd)
            weight = Variable(xp.asarray(weight * 1.0).astype('float32'))
            target = cwords[window + offset: -(window + 1) + offset]
            l = self._neg_sample(cntxt, target, weight)
            loss = l.data if loss is None else loss + l.data
            l.backward()
        return loss

    def _target(self, data_cat_feats, data_targets):
        """ Calculate the local losses relating individual document topic
        weights to the target prediction for those documents. Additionally
        calculate a regression on all documents to global targets.
        """
        losses = None
        weights = []
        args = (data_cat_feats, self.categorical_feature_names, data_targets)
        for data_cat_feat, cat_feat_name, data_target in zip(*args):
            cat_feat = self.categorical_features[cat_feat_name]
            (embedding, transform, loss_func, penalty, l2_penalty,
                logdet_penalty) = cat_feat
            weights.append(embedding.proportions(data_cat_feat, softmax=True))
            if loss_func is None:
                continue
            # This function will input an ID and ouput
            # (batchsize, n_hidden)
            latent = embedding.proportions(data_cat_feat, softmax=True)
            # Transform (batchsize, n_hidden) -> (batchsize, n_dim)
            # n_dim is 1 for RMSE, 1 for logistic outcomes, n for softmax
            output = F.dropout(transform(latent), ratio=self.dropout_ratio)
            # Loss_func gives likelihood of data_target given output
            shape = output.data.shape
            l = loss_func(output, F.reshape(data_target, shape))
            losses = l if losses is None else losses + l
        # Construct the latent vectors for all doc_ids
        if len(weights) > 0:
            feature_values = F.concat(weights)
        for name, (transform, loss_func, dtype) in self.target_losses.items():
            prediction = transform(feature_values)
            data_target = data_targets[name]
            l = loss_func(prediction, data_target)
            losses = l if losses is None else losses + l
        if losses is None:
            losses = Variable(self.xp.asarray(0.0, dtype='float32'))
        return losses

    def to_var(self, c):
        if 'float' in str(c.dtype):
            return Variable(self.xp.asarray(c.astype('float32')))
        else:
            return Variable(self.xp.asarray(c.astype('int32')))

    def _check_input(self, word_matrix, categorical_features, targets):
        if word_matrix is not None:
            word_matrix = word_matrix.astype('int32')
        if self._finalized is False:
            self.finalize()
        if isinstance(categorical_features, (np.ndarray, np.generic)):
            # If we pass in a single categorical feature, wrap it into a list
            categorical_features = [categorical_features]
        msg = "target variable must be of format {'target_name': nd.array}"
        assert not isinstance(targets, (np.ndarray, np.generic)), msg
        if categorical_features is None:
            categorical_features = []
        else:
            msg = "Number of categorical features not equal to initialized"
            test = len(categorical_features) == len(self.categorical_features)
            assert test, msg
            categorical_features = [self.to_var(c)
                                    for c in categorical_features]
        if targets is None:
            targets = []
        else:
            msg = "Number of targets not equal to initialized no. of targets"
            vals = self.categorical_features.values()
            assert len(targets) == sum([c[2] is not None for c in vals])
            targets = [self.to_var(c) for c in targets]
        for i, categorical_feature in enumerate(categorical_features):
            msg = "Number of rows in word matrix unequal"
            msg += "to that in categorical feature #%i" % i
            if word_matrix is not None:
                assert word_matrix.shape[0] == \
                    categorical_feature.data.shape[0], msg
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
                    targets=None, itr=None, n_itr=None):
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
        targets : dict of float or int arrays
            This is usually side information related to a document. Latent
            categorical_features are chosen so that targets will correlated
            with them. For example, this could be the sold outcome of a client
            comment or the number of votes a comment receives. The key of this
            dictionary must be a string matching the name given to the target
            loss function. The values must the outcome numpy arrays.
        itr : int
            Passed to fit_partial to be printed in the logger detailing which
            iteration we're currently on.
        n_itr : int
            Passed to fit_partial to be printed in the logger specifying the
            total number of iterations.
        """
        t0 = time.time()
        self._n_partial_fits += 1
        words_flat, vcategorical_features, targets = \
            self._check_input(words_flat, categorical_features, targets)
        self._update_comp_counts(categorical_features)
        # Before calculating gradients, zero them or we will get NaN
        self.zerograds()
        prior_loss = self._priors()
        words_loss = self._skipgram_flat(words_flat, categorical_features)
        trget_loss = self._target(vcategorical_features, targets)
        # Loss is composed of loss from predicting the word given context,
        # the target given the context, and the loss due to the prior
        # on the mixture embedding
        total_loss = trget_loss + prior_loss * fraction
        # Calculate back gradients
        total_loss.backward()
        # Propagate gradients
        self._optimizer.update()
        # Report loss, speed, timings
        t1 = time.time()
        rate = words_flat.shape[0] / (t1 - t0)
        msg = "Loss: %1.5e Prior: %1.5e Target %1.5e Rate: %1.2e wps"
        msg = msg % (words_loss, prior_loss.data * fraction,
                     trget_loss.data, rate)
        msg += " ETA: %1.1es" % ((n_itr - itr) * (t1 - t0))
        if itr is not None:
            msg += " Itr %i/%i" % (itr, n_itr)
        self.logger.info(msg)
        return total_loss

    def fit(self, words_flat, categorical_features=None, targets=None,
            epochs=10, fraction=None, n_chunk=None):
        """ Train the latent document-to-topic weights, topic vectors,
        and word vectors on the full dataset.

        Arguments
        ---------
        words_flat : int array
            A flattened 1D array of shape (n_observations) where each row is
            in a single document.
        fraction : float
            Break the data into chunks of this fractional size. Must define
            this or the batchsize.
        n_chunk : int
            The size of the individual minibatches. Usually, this number is
            increased to maximize GPU efficiency.
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
        msg = "Either fraction or n_chunk must be specified"
        assert (fraction is not None) or (n_chunk is not None), msg
        if n_chunk is None:
            n_chunk = int(words_flat.shape[0] * fraction)
        if fraction is None:
            fraction = n_chunk * 1.0 / words_flat.shape[0]
        self.logger.info("Set chunk size to {:d}".format(n_chunk))
        self.logger.info("Set fraction to {:1.3e}".format(fraction))
        for epoch in range(epochs):
            args = []
            if categorical_features is not None:
                args += categorical_features
            n_cat_feats = len(args)
            if targets is not None:
                args += targets
            for j, dat in enumerate(_chunks(n_chunk, words_flat, *args)):
                chunk = dat.pop(0)
                cat_feat, target = dat[:n_cat_feats], dat[n_cat_feats:]
                this_fraction = len(chunk) * 1.0 / words_flat.shape[0]
                self.fit_partial(chunk, this_fraction,
                                 categorical_features=cat_feat, targets=target,
                                 itr=j, n_itr=int(1.0 / fraction))

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
            print prefix + ' '.join([data['vocab'][i].strip().replace(' ', '_')
                                    for i in top])


def _chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    keypoints = []
    for i in xrange(0, len(args[0]), n):
        keypoints.append((i, i + n))
    random.shuffle(keypoints)
    for a, b in keypoints:
        yield [arg[a: b] for arg in args]
