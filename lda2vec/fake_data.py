import numpy as np
from numpy.random import random_sample


def orthogonal_matrix(shape):
    # Stolen from blocks:
    # github.com/mila-udem/blocks/blob/master/blocks/initialization.py
    M1 = np.random.randn(shape[0], shape[0])
    M2 = np.random.randn(shape[1], shape[1])

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))

    n_min = min(shape[0], shape[1])
    return np.dot(Q1[:, :n_min], Q2[:n_min, :])


def softmax(w):
    # https://gist.github.com/stober/1946926
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)[:, None]
    return dist


def sample(values, probabilities, size):
    assert np.allclose(np.sum(probabilities, axis=-1), 1.0)
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]


def fake_data(n_docs, n_words, n_sent_length, n_topics):
    """ Generate latent topic vectors for words and documents
    and then for each document, draw a sentence. Draw each word
    document with probability proportional to the dot product and
    normalized with a softmax.

    Arguments
    ---------
    n_docs : int
        Number of documents
    n_words : int
        Number of words in the vocabulary
    n_sent_length : int
        Number of words to draw for each document
    n_topics : int
        Number of topics that a single document can belong to.

    Returns
    -------
    sentences : int array
        Array of word indices of shape (n_docs, n_sent_length).

    """
    # These are log ratios for the doc & word topics
    doc_topics = orthogonal_matrix([n_docs, n_topics])
    wrd_topics = orthogonal_matrix([n_topics, n_words])
    # Multiply log ratios and softmax to get prob of word in doc
    doc_to_wrds = softmax(np.dot(doc_topics, wrd_topics))
    # Now sample from doc_to_wrd to get realizations
    indices = np.arange(n_words).astype('int32')
    sentences = []
    for doc_to_wrd in doc_to_wrds:
        words = sample(indices, doc_to_wrd, n_sent_length)
        sentences.append(words)
    sentences = np.array(sentences)
    return sentences.astype('int32')
