from collections import defaultdict
import numpy as np


class Corpus():
    def __init__(self, out_of_vocabulary=-2):
        """ The Corpus helps with tasks involving integer representations of
        words. This object is used to filter, subsample, and convert loose
        word indices to compact word indices.

        'Loose' word arrays are word indices given by a tokenizer. The word
        index is not necessarily representative of word's frequency rank, and
        so loose arrays tend to have 'gaps' of unused indices, which can make
        models less memory efficient. As a result, this class helps convert
        a loose array to a 'compact' one where the most common words have low
        indices, and the most infrequent have high indices.

        Corpus maintains a count of how many of each word it has seen so
        that it can later selectively filter frequent or rare words. However,
        since word popularity rank could change with incoming data the word
        index count must be updated fully and `self.finalize()` must be called
        before any filtering and subsampling operations can happen.

        >>> corpus = Corpus()
        >>> words_raw = np.arange(25).reshape((5, 5))
        >>> corpus.update_word_count(words_raw)
        >>> corpus.finalize()
        >>> word_compact = corpus.to_compact(words_raw)
        >>> words_pruned = corpus.filter_count(words_compact, min_count=20)
        >>> words_sub = corpus.subsample_frequent(words_pruned, thresh=1e-5)
        >>> word_loose = corpus.covnert_to_loose(word_sub)
        >>> np.all(word_loose == words_raw)
        """
        self.counts_loose = defaultdict(int)
        self._finalized = False
        self.out_of_vocabulary = out_of_vocabulary

    def update_word_count(self, loose_array):
        """ Update the corpus word counts given a loose array of word indices.
        Can be called multiple times, but once `finalize` is called the word
        counts cannot be updated.

        Arguments
        ---------
        loose_array : int array
            Array of word indices.

        >>> corpus = Corpus()
        >>> corpus.update_word_count(np.arange(10))
        >>> corpus.update_word_count(np.arange(8))
        >>> corpus.counts_loose[0]
        2
        >>> corpus.counts_loose[9]
        1
        """
        self._check_unfinalized()
        uniques, counts = np.unique(np.ravel(loose_array), return_counts=True)
        for k, v in zip(uniques, counts):
            self.counts_loose[k] += v

    def finalize(self):
        """ Call `finalize` once done updating word counts. This means the
        object will no longer accept new word count data, but the loose
        to compact index mapping can be computed. This frees the object to
        filter, subsample, and compactify incoming word arrays.

        >>> corpus = Corpus()

        We'll update the word counts, making sure that word index 2
        is the most common word index.
        >>> corpus.update_word_count(np.arange(1) + 2)
        >>> corpus.update_word_count(np.arange(3) + 2)
        >>> corpus.update_word_count(np.arange(10) + 2)
        >>> corpus.update_word_count(np.arange(8) + 2)
        >>> corpus.counts_loose[2]
        4

        The corpus has not been finalized yet, and so the compact mapping
        has not yet been computed.
        >>> corpus.counts_compact[0]
        Traceback (most recent call last):
            ...
        AttributeError: Corpus instance has no attribute 'counts_compact'
        >>> corpus.finalize()
        >>> corpus.counts_compact[0]
        4
        >>> corpus.loose_to_compact[2]
        0
        >>> corpus.loose_to_compact[3]
        2
        """
        carr = sorted(self.counts_loose.items(), key=lambda x: x[1],
                      reverse=True)
        keys = np.array(carr)[:, 0]
        cnts = np.array(carr)[:, 1]
        order = np.argsort(cnts)[::-1].astype('int32')
        loose_cnts = cnts[order]
        loose_keys = keys[order]
        compact_keys = np.arange(keys.shape[0]).astype('int32')
        loose_to_compact = {l: c for l, c in zip(loose_keys, compact_keys)}
        self.loose_to_compact = loose_to_compact
        self.counts_compact = {loose_to_compact[l]: c for l, c in
                               zip(loose_keys, loose_cnts)}
        self._counts_loose_arrs = dict(keys=loose_keys, counts=loose_cnts)
        self._finalized = True

    def _check_finalized(self):
        msg = "self.finalized() must be called before any other array ops"
        assert self._finalized, msg

    def _check_unfinalized(self):
        msg = "Cannot update word counts after self.finalized()"
        msg += "has been called"
        assert not self._finalized, msg

    def filter_count(self, arr, max_count=0, min_count=20000, pad=-1):
        """ Replace word indices below min_count with the pad index.

        Arguments
        ---------
        arr : int array
            Source array whose values will be replaced
        pad : int
            Rare word indices will be replaced with this index
        min_count : int
            Replace words less frequently occuring than this count. This
            defines the threshold for what words are very rare
        max_count : int
            Replace words occuring more frequently than this count. This
            defines the threshold for very frequent words

        >>> corpus = Corpus()

        Make 1000 word indices with index < 100 and update the word counts.
        >>> word_indices = np.random.randint(100, size=1000)
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()  # any word indices above 99 will be filtered

        Now create a new text, but with some indices above 100
        >>> word_indices = np.random.randint(200, size=1000)
        >>> word_indices.max() < 100
        False

        Remove words that have never appeared in the original corpus.
        >>> filtered = corpus.filter_count(word_indices, min_count=1)
        >>> filtered.max() < 100
        True

        We can also remove highly frequent words.
        >>> filtered = corpus.filter_count(word_indices, max_count=2)
        >>> len(np.unique(word_indices)) > len(np.unique(filtered))
        True
        """
        self._check_finalized()
        keys = self.loose_counts['keys'].copy()
        reps = self.loose_counts['keys'].copy()
        idx = np.ones_like(self.cnts_loose).astype('bool')
        if min_count:
            idx &= self.cnts_loose < min_count
        if max_count:
            idx &= self.cnts_loose > max_count
        reps[idx] = self.out_of_vocabulary
        ret = fast_replace(arr, keys, reps)
        return ret

    def subsample_frequent(self, arr, pad=-1, threshold=1e-5):
        """ Subsample the most frequent words. This aggressively
        drops word with frequencies higher than `threshold`.

        .. math :: p(w) = 1.0 - \sqrt{\frac{threshold}{f(w)}}

        .. [1] Distributed Representations of Words and Phrases and
               their Compositionality. Mikolov, Tomas and Sutskever, Ilya
               and Chen, Kai and Corrado, Greg S and Dean, Jeff
               Advances in Neural Information Processing Systems 26
        """
        self._check_finalized()
        raise NotImplemented

    def to_compact(self, word_loose):
        """ Convert a loose word index matrix to a compact array using
        a fixed loose to dense mapping. Out of vocabulary word indices
        will be replaced by the out of vocabulary index. The most common
        index will be mapped to 0, the next most common to 1, and so on.

        >>> corpus = Corpus()
        >>> word_indices = np.random.randint(100, size=1000)
        >>> corpus.update_word_count(word_indices)
        >>> corpus.finalize()  # any word indices above 99 will be filtered
        >>> word_compact = corpus.to_compact(word_indices)

        The most common word in the training set will be mapped to zero.
        >>> most_common = np.argmax(np.bincount(word_indices))
        >>> corpus.loose_to_compact[most_common] == 0
        True

        Out of vocabulary indices will be mapped to -1.
        >>> word_indices = np.random.randint(150, size=1000)
        >>> word_compact = corpus.to_compact(word_indices)
        >>> -1 in word_compact
        True
        """
        self._check_finalized()
        keys = self._counts_loose_arrs['keys'].copy()
        reps = self._counts_loose_arrs['keys'].copy()
        uniques = np.unique(word_loose)
        # Find the out of vocab indices
        oov = np.setdiff1d(uniques, keys, assume_unique=True)
        reps = np.concatenate((keys, np.zeros_like(oov) - 1))
        keys = np.concatenate((keys, oov))
        compact = fast_replace(word_loose, keys, reps)
        return compact

    def convert_to_loose(self, arr):
        self._check_finalized()
        raise NotImplemented


def fast_replace(data, keys, values, skip_checks=False):
    """ Do a search-and-replace in array `data`.

    Arguments
    ---------
    data : int array
        Array of integers
    keys : int array
        Array of keys inside of `data` to be replaced
    values : int array
        Array of values that replace the `keys` array
    skip_checks : bool, default=False
        Optionally skip sanity checking the input.

    >>> fast_replace(np.arange(5), np.arange(5), np.arange(5)[::-1])
    array([4, 3, 2, 1, 0])
    """
    assert np.allclose(keys.shape, values.shape)
    if not skip_checks:
        assert data.max() <= keys.max()
    sdx = np.argsort(keys)
    keys, values = keys[sdx], values[sdx]
    idx = np.digitize(data, keys, right=True)
    new_data = values[idx]
    return new_data
