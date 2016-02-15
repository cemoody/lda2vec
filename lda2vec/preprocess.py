import os
from spacy.en import English, LOCAL_DATA_DIR
from spacy.attrs import LOWER, LIKE_URL, LIKE_EMAIL

import numpy as np

nlp = None


def tokenize(texts, max_length, skip=-2, attr=LOWER, merge=False, **kwargs):
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
    skip : int, optional
        Short documents will be padded with this variable up until max_length.
    attr : int, from spacy.attrs
        What to transform the token to. Choice must be in spacy.attrs, and =
        common choices are (LOWER, LEMMA)
    merge : int, optional
        Merge noun phrases into a single token. Useful for turning 'New York'
        into a single token.
    kwargs : dict, optional
        Any further argument will be sent to the spaCy tokenizer. For extra
        speed consider setting tag=False, parse=False, entity=False.

    Returns
    -------
    arr : 2D array of ints
        Has shape (len(texts), max_length). Each value represents
        the word index.
    vocab : dict
        Keys are the word index, and values are the string. The pad index gets
        mapped to None

    >>> sents = [u"Do you recall", u"not long ago a@b.com", u"hello zombo.com"]
    >>> arr, vocab = tokenize(sents, 10)
    >>> arr.shape[0]
    3
    >>> arr.shape[1]
    10
    >>> w2i = {w: i for i, w in vocab.iteritems()}
    >>> arr[0, 0] == w2i[u'do']  # First word and its index should match
    True
    >>> arr[0, -1]  # last word in 0th document is a pad word
    -2
    >>> arr[1, 2] == w2i[u'ago']
    True
    >>> arr[1, 3]  # The email token is thrown out
    -2
    >>> arr[2, 1]  # The URL token is thrown out
    -2
    """
    global nlp
    if nlp is None:
        data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
        nlp = English(data_dir=data_dir)
    data = np.zeros((len(texts), max_length), dtype='int32')
    data[:] = skip
    bad_deps = ('amod', 'compound')
    for row, doc in enumerate(nlp.pipe(texts, n_threads=4)):
        if merge:
            # from the spaCy blog, an example on how to merge
            # noun phrases into single tokens
            for phrase in doc.noun_chunks:
                # Only keep adjectives and nouns, e.g. "good ideas"
                while len(phrase) > 1 and phrase[0].dep_ not in bad_deps:
                    phrase = phrase[1:]
                if len(phrase) > 1:
                    # Merge the tokens, e.g. good_ideas
                    phrase.merge(phrase.root.tag_, phrase.text,
                                 phrase.root.ent_type_)
                # Iterate over named entities
                for ent in doc.ents:
                    if len(ent) > 1:
                        # Merge them into single tokens
                        ent.merge(ent.root.tag_, ent.text, ent.label_)
        dat = doc.to_array([attr, LIKE_EMAIL, LIKE_URL]).astype('int32')
        msg = "Negative indices reserved for special tokens"
        assert dat.min() >= 0, msg
        # Replace email and URL tokens
        idx = (dat[:, 1] > 0) | (dat[:, 2] > 0)
        dat[idx] = skip
        length = min(len(dat), max_length)
        data[row, :length] = dat[:length, 0].ravel()
    uniques = np.unique(data)
    vocab = {v: nlp.vocab[v].lower_ for v in uniques if v != skip}
    vocab[skip] = '<SKIP>'
    return data, vocab


if __name__ == "__main__":
    import doctest
    doctest.testmod()
