import os
from spacy.en import English, LOCAL_DATA_DIR
from spacy.attrs import LOWER

import numpy as np

nlp = None


def tokenize(texts, max_length, pad=-1):
    """ Uses spaCy to quickly tokenize text and return an array
    of indices.

    This method stores a global NLP directory in memory, and takes
    up to a minute to run for the time. Later calls will have the
    tokenizer in memory.

    Parameters
    ----------
    text : list of unicode strings
        These are the input documents. There can be multiple sentences per
        item in the list.
    max_length : int
        This is the maximum number of words per document. If the document is
        shorter then this number it will be padded to this length.
    pad : int, optional
        Short documents will be padded with this variable up until max_length.

    Returns
    -------
    arr : 2D array of ints
        Has shape (len(texts), max_length). Each value represents
        the word index.
    counts : 1D array of ints
        The number of times each of the unique values comes up in the original
        array.
    vocab : dict
        Keys are the word index, and values are the string. The pad index gets
        mapped to None

    >>> arr, counts, vocab = tokenize([u"Do you recall", u"not long ago"], 5)
    >>> w2i = {w: i for i, w in vocab.iteritems()}
    >>> arr[0, 0] == w2i[u'Do']  # First word and its index should match
    True
    >>> arr[0, -1]  # last word in 0th document is a pad word
    -1
    """
    global nlp
    if nlp is None:
        data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
        nlp = English(data_dir=data_dir)
    data = np.zeros((len(texts), max_length), dtype='int32')
    data[:] = pad
    for row, text in enumerate(texts):
        doc = nlp(text)
        dat = doc.to_array([LOWER]).astype('int32')
        length = min(len(dat), max_length)
        data[row, :length] = dat[:length, :].ravel()
    uniques, counts = np.unique(data, return_counts=True)
    vocab = {v: nlp.vocab[v].orth_ for v in uniques if v != -1}
    msg = "Cannot pick a pad integer that is legal spaCy word index"
    assert pad not in vocab, msg
    vocab[pad] = None
    return data, counts, vocab

if __name__ == "__main__":
    import doctest
    doctest.testmod()
