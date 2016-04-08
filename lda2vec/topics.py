import numpy as np


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def _softmax_2d(x):
    y = x - x.max(axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y


def prob_words(context, vocab, temperature=1.0):
    """ This calculates a softmax over the vocabulary as a function
    of the dot product of context and word.
    """
    dot = np.dot(vocab, context)
    prob = _softmax(dot / temperature)
    return prob


def prepare_topics(weights, factors, word_vectors, vocab, temperature=1.0,
                   doc_lengths=None, term_frequency=None, normalize=False):
    """ Collects a dictionary of word, document and topic distributions.

    Arguments
    ---------
    weights : float array
        This must be an array of unnormalized log-odds of document-to-topic
        weights. Shape should be [n_documents, n_topics]
    factors : float array
        Should be an array of topic vectors. These topic vectors live in the
        same space as word vectors and will be used to find the most similar
        words to each topic. Shape should be [n_topics, n_dim].
    word_vectors : float array
        This must be a matrix of word vectors. Should be of shape
        [n_words, n_dim]
    vocab : list of str
        These must be the strings for words corresponding to
        indices [0, n_words]
    temperature : float
        Used to calculate the log probability of a word. Higher
        temperatures make more rare words more likely.
    doc_lengths : int array
        An array indicating the number of words in the nth document.
        Must be of shape [n_documents]. Required by pyLDAvis.
    term_frequency : int array
        An array indicating the overall number of times each token appears
        in the corpus. Must be of shape [n_words]. Required by pyLDAvis.

    Returns
    -------
    data : dict
        This dictionary is readily consumed by pyLDAVis for topic
        visualization.
    """
    # Map each factor vector to a word
    topic_to_word = []
    msg = "Vocabulary size did not match size of word vectors"
    assert len(vocab) == word_vectors.shape[0], msg
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis=1)[:, None]
    # factors = factors / np.linalg.norm(factors, axis=1)[:, None]
    for factor_vector in factors:
        factor_to_word = prob_words(factor_vector, word_vectors,
                                    temperature=temperature)
        topic_to_word.append(np.ravel(factor_to_word))
    topic_to_word = np.array(topic_to_word)
    msg = "Not all rows in topic_to_word sum to 1"
    assert np.allclose(np.sum(topic_to_word, axis=1), 1), msg
    # Collect document-to-topic distributions, e.g. theta
    doc_to_topic = _softmax_2d(weights)
    msg = "Not all rows in doc_to_topic sum to 1"
    assert np.allclose(np.sum(doc_to_topic, axis=1), 1), msg
    data = {'topic_term_dists': topic_to_word,
            'doc_topic_dists': doc_to_topic,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency}
    return data


def print_top_words_per_topic(data, top_n=10, do_print=True):
    """ Given a pyLDAvis data array, print out the top words in every topic.

    Arguments
    ---------
    data : dict
        A dict object that summarizes topic data and has been made using
        `prepare_topics`.
    """
    msgs = []
    for j, topic_to_word in enumerate(data['topic_term_dists']):
        top = np.argsort(topic_to_word)[::-1][:top_n]
        prefix = "Top words in topic %i " % j
        msg = ' '.join([data['vocab'][i].strip().replace(' ', '_')
                        for i in top])
        if do_print:
            print prefix + msg
        msgs.append(msg)
    return msgs
