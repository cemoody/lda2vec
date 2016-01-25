from lda2vec import preprocess
import numpy as np
import pytest
import os

on_ci = os.environ.get('CI', False) == 'true'


@pytest.mark.skipif(on_ci, reason='SpaCy install fails on TravisCI')
def test_tokenize():
    texts = [u'Do you recall, not long ago']
    texts += [u'We would walk on the sidewalk?']
    arr, vocab = preprocess.tokenize(texts, 10)
    assert arr[0, 0] != arr[0, 1]
    assert arr.shape[0] == 2
    assert arr.shape[1] == 10
    assert arr[0, -1] == -2
    assert arr.dtype == np.dtype('int32')
    first_word = texts[0].split(' ')[0].lower()
    first_lowr = preprocess.nlp.vocab[arr[0, 0]].lower_
    assert first_word == first_lowr
