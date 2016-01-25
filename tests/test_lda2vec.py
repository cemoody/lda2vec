from lda2vec import LDA2Vec
from lda2vec import fake_data
import numpy as np

# TODO: Test for component w/ & w/o loss type
# TODO: Test over all loss types
# TODO: Test prior / word / target losses
# TODO: Test raise error for zero components
# TODO: Test raise error for uninitialized components
# TODO: Test for visualization methods


def generate(n_docs=300, n_words=100, n_sent_length=5, n_hidden=8):
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
    model.add_categorical_feature(n_docs, n_topics)
    log_perp = model.compute_log_perplexity(words,
                                            categorical_features=[doc_ids])
    log_prob = np.log(1.0 / n_wrds)
    theoretical = log_prob / n_obs
    msg = "Initial log perplexity is significantly different from uniform"
    diff = np.abs(log_perp.data - theoretical) / theoretical
    assert diff < 1e-3, msg


def categorical_feature(partial=True, name=None, n_cat_feat=2, itrs=15):
    n_topics = 2
    model, words, doc_ids = generate()
    n_docs = doc_ids.max() + 1
    categorical_features = []
    for j in range(n_cat_feat):
        if name:
            name += str(j)
        model.add_categorical_feature(n_docs, n_topics, name=name)
        categorical_features.append(doc_ids)
    cf = categorical_features
    perp_orig = model.compute_log_perplexity(words, categorical_features=cf)
    # Increase learning rate
    # model._optimizer.alpha = 1e-1
    # Test perplexity decreases
    if partial:
        for _ in range(itrs):
            model.fit_partial(words, 1.0, categorical_features=cf)
    else:
        for _ in range(itrs):
            model.fit(words, categorical_features=cf)
    perp_fit = model.compute_log_perplexity(words, categorical_features=cf)
    msg = "Perplexity should improve with a fit model"
    assert perp_fit.data < perp_orig.data, msg
    del model


def test_single_categorical_feature_no_name():
    categorical_feature(n_cat_feat=1)
    categorical_feature(n_cat_feat=1, partial=True)


def test_single_categorical_feature_named():
    categorical_feature(n_cat_feat=1, name="named_layer", itrs=100)
    categorical_feature(n_cat_feat=1, name="named_layer", partial=True,
                        itrs=100)


def test_multiple_categorical_features_no_names():
    categorical_feature(n_cat_feat=3)
    categorical_feature(n_cat_feat=3, partial=True)


def test_multiple_categorical_features_named():
    categorical_feature(n_cat_feat=3, name="named_layer")
    categorical_feature(n_cat_feat=3, name="named_layer", partial=True)


def entropy(p):
    return -np.nansum((p + 1e-12) * np.log(p + 1e-12))


def exp_entropy(log_p):
    return -np.nansum(np.exp(log_p + 1e-12) * (log_p + 1e-12))


def test_log_prob_words():
    n_topics = 2
    model, words, doc_ids = generate()
    n_docs = doc_ids.max() + 1
    model.add_categorical_feature(n_docs, n_topics)
    comp = np.zeros(1).astype('int32')
    low = model.log_prob_words(comp, temperature=1e-2).data
    high = model.log_prob_words(comp, temperature=1e6).data
    msg = "Lower temperatures should be lower entropy and more concentrated"
    assert exp_entropy(low) < exp_entropy(high), msg


def prepare_topics():
    n_topics = 8
    model, words, doc_ids = generate(n_hidden=n_topics)
    vocab = ["%i" % i for i in range(words.max() + 1)]
    n_docs = doc_ids.max() + 1
    model.add_categorical_feature(n_docs, n_topics, name='catfeat')
    model.finalize()
    model.fit(words, categorical_features=[doc_ids])
    return model, words, vocab


def prepare_topics_checks(model, data):
    catfeat, _, _ = model.categorical_features.values()[0]
    cols = {'topic_term_dists': [catfeat.n_topics, model.n_words],
            'doc_topic_dists': [catfeat.n_documents, catfeat.n_topics],
            'doc_lengths': [catfeat.n_documents],
            'vocab': [model.n_words],
            'term_frequency': [model.n_words]}
    for col, shape in cols.items():
        msg = "Missing column '%s' in data" % col
        assert col in data, msg
        msg = "Shape of prepared data did not macth expected shape"
        datum = np.array(data[col])
        assert np.allclose(shape, datum.shape), msg
    msg = "Document lengths should be nonzero if model has been fit"
    assert data['doc_lengths'].sum() >= 0, msg


def prepare_topics_numbered_named(featname='catfeat'):
    model, words, vocab = prepare_topics()
    data = model.prepare_topics(featname, vocab, temperature=1.0)
    prepare_topics_checks(model, data)


def test_prepare_topics_numbered_named():
    for featname in [0, 'catfeat']:
        prepare_topics_numbered_named(featname)
