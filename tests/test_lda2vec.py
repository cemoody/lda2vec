from lda2vec import LDA2Vec
import numpy as np


def test_no_component():
    n_words = 10
    n_docs = 300
    n_sent_length = 5
    n_hidden = 8
    words = np.random.randint(n_words, size=(n_docs, n_sent_length))
    _, counts = np.unique(words, return_counts=True)
    model = LDA2Vec(n_words, n_sent_length, n_hidden, counts)
    model.fit_partial(words, 1.0)


def test_single_component():
    n_words = 10
    n_docs = 300
    n_sent_length = 5
    n_hidden = 8
    n_topics = 2
    words = np.random.randint(n_words, size=(n_docs, n_sent_length))
    doc_ids = np.arange(n_docs)
    _, counts = np.unique(words, return_counts=True)
    model = LDA2Vec(n_words, n_sent_length, n_hidden, counts)
    model.add_component(n_docs, n_topics)
    model.fit_partial(words, 1.0, contexts=[doc_ids])


def test_multiple_component():
    pass


def test_single_component_target():
    # Test G/CPU
    pass


def test_visualization():
    pass

# Test for component w/ & w/o loss type
# Test over all loss types
# Test prior / word / target losses
