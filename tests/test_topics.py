from lda2vec import topics

import numpy as np


def exp_entropy(log_p):
    return -np.nansum(np.exp(log_p + 1e-12) * (log_p + 1e-12))


def test_prob_words():
    context = np.random.randn(3)
    vocab = np.random.randn(10, 3)
    lo = topics.prob_words(context, vocab, temperature=1)
    hi = topics.prob_words(context, vocab, temperature=1e6)
    msg = "Lower temperatures should be lower entropy and more concentrated"
    assert exp_entropy(np.log(lo)) < exp_entropy(np.log(hi)), msg


def prepare_topics():
    # One document in two topics, unnormalized
    weights = np.array([[0.5, -0.1]])
    # Two topics in 4 dimensions
    factors = np.array([[0.1, 0.1, 0.1, 5.0],
                        [5.1, 0.1, 0.1, 0.0]])
    # Three words in 4 dimensions
    vectors = np.array([[5.0, 0.1, 0.1, 0.1],
                        [0.0, 0.1, 0.1, 5.0],
                        [2.0, 0.1, 0.1, -.9]])
    vocab = ['a', 'b', 'c']
    data = topics.prepare_topics(weights, factors, vectors, vocab)
    return data


def test_prepare_topics():
    data = prepare_topics()
    t2w = data['topic_term_dists']
    msg = "Topic 0 should be most similar to 2nd token"
    assert t2w[0].argsort()[::-1][0] == 1, msg
    msg = "Topic 1 should be most similar to 1st token"
    assert t2w[1].argsort()[::-1][0] == 0, msg


def test_print_top_words_per_topic():
    data = prepare_topics()
    msgs = topics.print_top_words_per_topic(data, do_print=False)
    assert len(msgs) == 2
    for msg in msgs:
        assert len(msg.split(' ')) == 3
