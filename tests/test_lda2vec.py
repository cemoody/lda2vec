from lda2vec import LDA2Vec
from lda2vec import fake_data
import numpy as np

# TODO: Test for component w/ & w/o loss type
# TODO: Test over all loss types
# TODO: Test prior / word / target losses
# TODO: Test raise error for zero components
# TODO: Test raise error for uninitialized components
# TODO: Test for visualization methods


def generate(n_docs=300, n_words=10, n_sent_length=5, n_hidden=8):
    words = fake_data.fake_data(n_docs, n_words, n_sent_length, n_hidden)
    words_flat = words.ravel()
    doc_ids = np.repeat(np.arange(words.shape[0]).astype('int32'),
                        n_sent_length)
    doc_ids = doc_ids.ravel()
    _, counts = np.unique(words_flat, return_counts=True)
    model = LDA2Vec(n_words, n_hidden, counts, n_samples=1)
    return model, words_flat, doc_ids


def test_compute_perplexity():
    model, words, doc_ids = generate()
    n_topics = 2
    n_docs = words.shape[0]
    n_wrds = words.max() + 1
    n_obs = np.prod(words.shape)
    doc_ids = np.arange(n_docs)
    model.add_component(n_docs, n_topics)
    log_perp = model.compute_log_perplexity(words, components=[doc_ids])
    log_prob = np.log(1.0 / n_wrds)
    theoretical = log_prob / n_obs
    msg = "Initial log perplexity is significantly different from uniform"
    diff = np.abs(log_perp.data - theoretical) / theoretical
    assert diff < 1e-3, msg


def component(model, words, name=None, n_comp=1, itrs=5):
    n_topics = 2
    model, words, doc_ids = generate()
    n_docs = doc_ids.max() + 1
    components = []
    for j in range(n_comp):
        if name:
            name += str(j)
        model.add_component(n_docs, n_topics, name=name)
        components.append(doc_ids)
    perp_orig = model.compute_log_perplexity(words, components=components)
    # Increase learning rate
    # model._optimizer.alpha = 1e-1
    # Test perplexity decreases
    for _ in range(itrs):
        model.fit_partial(words, 1.0, components=components)
    perp_fit = model.compute_log_perplexity(words, components=components)
    msg = "Perplexity should improve with a fit model"
    assert perp_fit.data < perp_orig.data, msg
    del model


def test_single_component_no_name():
    model, words, doc_ids = generate()
    component(model, words)


def test_single_component_named():
    model, words, doc_ids = generate()
    component(model, words, name="named_layer")


def test_multiple_components_no_names():
    model, words, doc_ids = generate()
    component(model, words, n_comp=3)


def test_multiple_components_named():
    model, words, doc_ids = generate()
    component(model, words, name="named_layer", n_comp=3)
