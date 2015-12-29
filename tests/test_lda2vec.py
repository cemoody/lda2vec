from lda2vec import LDA2Vec
import numpy as np

# TODO: Test for component w/ & w/o loss type
# TODO: Test over all loss types
# TODO: Test prior / word / target losses
# TODO: Test raise error for zero components
# TODO: Test raise error for uninitialized components
# TODO: Test for visualization methods


def generate(n_docs=300, n_words=10, n_sent_length=5, n_hidden=8):
    words = np.random.randint(n_words, size=(n_docs, n_sent_length))
    _, counts = np.unique(words, return_counts=True)
    model = LDA2Vec(n_words, n_sent_length, n_hidden, counts)
    return model, words


def test_compute_perplexity():
    model, words = generate()
    n_docs = words.shape[0]
    n_wrds = words.max() + 1
    n_obs = np.prod(words.shape)
    doc_ids = np.arange(n_docs)
    log_perp = model.compute_log_perplexity(words, components=[doc_ids])
    log_prob = np.log(1.0 / n_wrds)
    theoretical = log_prob / n_obs
    msg = "Initial log perplexity is significantly different from uniform"
    diff = np.abs(log_perp - theoretical) / theoretical
    assert diff < 1e-3, msg


def component(model, words, name=None, n_comp=1):
    n_topics = 2
    n_docs = words.shape[0]
    doc_ids = np.arange(n_docs)
    model = generate()
    perp_orig = model.compute_log_perplexity(words, components=[doc_ids])
    for j in range(n_comp):
        if name:
            name += str(j)
        model.add_component(n_docs, n_topics, name=name)
    # Test perplexity decreases
    model.fit_partial(words, 1.0, components=[doc_ids])
    perp_fit = model.compute_log_perplexity(words, components=[doc_ids])
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
