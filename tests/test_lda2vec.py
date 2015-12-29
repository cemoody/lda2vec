from lda2vec import LDA2Vec
import numpy as np


def generate(n_docs=300, n_words=10, n_sent_length=5, n_hidden=8):
    words = np.random.randint(n_words, size=(n_docs, n_sent_length))
    _, counts = np.unique(words, return_counts=True)
    model = LDA2Vec(n_words, n_sent_length, n_hidden, counts)
    return model, words


def component(model, words, name=None, n_comp=1):
    n_topics = 2
    n_docs = words.shape[0]
    doc_ids = np.arange(n_docs)
    model = generate()
    perp_orig = model.calculate_log_perplexity(words, components=[doc_ids])
    for j in range(n_comp):
        if name:
            name += str(j)
        model.add_component(n_docs, n_topics, name=name)
    # Test perplexity decreases
    model.fit_partial(words, 1.0, components=[doc_ids])
    perp_fit = model.calculate_log_perplexity(words, components=[doc_ids])
    msg = "Perplexity should improve with a fit model"
    assert perp_fit < perp_orig, msg
    del model


def test_single_component_no_name():
    model, words = generate()
    component(model, words)


def test_single_component_named():
    model, words = generate()
    component(model, words, name="named_layer")


def test_multiple_components_no_names():
    model, words = generate()
    component(model, words, n_comp=2)


def test_multiple_components_named():
    model, words = generate()
    component(model, words, name="named_layer", n_comp=2)


def test_visualization():
    pass

# Test for component w/ & w/o loss type
# Test over all loss types
# Test prior / word / target losses
# Test raise error for zero components
# Test raise error for uninitialized components
