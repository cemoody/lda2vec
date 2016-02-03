from lda2vec import fake_data
from chainer import Variable
from chainer.functions import cross_covariance
import numpy as np


def test_orthogonal_matrix():
    msg = "Orthogonal matrices have equal inverse and transpose"
    arr = fake_data.orthogonal_matrix([20, 20])
    assert np.allclose(np.linalg.inv(arr), arr.T), msg


def test_orthogonal_matrix_covariance():
    msg = "Orthogonal matrix should have less covariance than a random matrix"
    orth = Variable(fake_data.orthogonal_matrix([20, 20]).astype('float32'))
    rand = Variable(np.random.randn(20, 20).astype('float32'))
    orth_cc = cross_covariance(orth, orth).data
    rand_cc = cross_covariance(rand, rand).data
    assert orth_cc < rand_cc, msg


def test_softmax():
    arr = np.random.randn(100, 15)
    probs = fake_data.softmax(arr)
    norms = np.sum(probs, axis=1)
    assert np.allclose(norms, np.ones_like(norms))


def test_sample():
    n_categories = 10
    idx = 4
    probs = np.zeros(n_categories)
    probs = np.array(probs)
    probs[idx] = 1.0
    values = np.arange(n_categories)
    size = 10
    draws = fake_data.sample(values, probs, size)
    assert np.all(draws == idx)


def test_fake_data():
    n_docs = 100
    n_words = 10
    n_hidden = 2
    n_sent_length = 5
    data = fake_data.fake_data(n_docs, n_words, n_sent_length, n_hidden)
    assert data.dtype == np.dtype('int32')
    assert data.shape[0] == n_docs
    assert data.shape[1] == n_sent_length
    assert np.max(data) <= n_words - 1
