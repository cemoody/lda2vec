from collections import defaultdict
import numpy as np


class Corpus():
    def __init__(self, counts=None):
        """ The corpus helps with tasks involving integer representations of
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
        >>> corpus.update_word_count(words_raw)
        >>> corpus.finalize()
        >>> words_pruned = corpus.filter_count(words_raw, min_count=20)
        >>> words_sub = corpus.subsample_frequent(words_pruned, thresh=1e-5)
        >>> word_compact = corpus.convert_to_compact(words_sub)
        >>> word_loose = corpus.covnert_to_loose(word_compact)
        >>> np.all(word_loose == words_sub)
        """
        self.counts = defaultdict(int)
        self._finalized = False

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
        >>> corpus.counts[0]
        2
        >>> corpus.counts[9]
        1
        """
        self._check_unfinalized()
        uniques, counts = np.unique(np.ravel(loose_array), return_counts=True)
        for k, v in zip(uniques, counts):
            self.counts[k] += v

    def finalize(self):
        """ Call `finalize` once done updating word counts. This means the
        object will no longer accept new word count data, but is free to
        filter, subsample, and compactify incoming word arrays.

        >>> corpus = Corpus()
        >>> corpus.update_word_count(np.arange(10))
        >>> corpus.update_word_count(np.arange(8))
        >>> corpus.finalize()
        >>> corpus._check_finalized()
        """
        self._finalized = True

    def _check_finalized(self):
        msg = "self.finalized() must be called before any other array ops"
        assert self._finalized, msg

    def _check_unfinalized(self):
        msg = "Cannot update word counts after self.finalized()"
        msg += "has been called"
        assert self._finalized, msg

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
        """
        self._check_finalized()
        raise NotImplemented

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

    def convert_to_compact(self, arr):
        self._check_finalized()
        raise NotImplemented

    def convert_to_loose(self, arr):
        self._check_finalized()
        rep_keys = np.arange(len(counts)).astype('int32')
        rep_vals = np.argsort(counts)[::-1].astype('int32')
        idx = np.digitize(data, rep_keys, right=True)
        new_data = rep_vals[idx]
        old2new = {k: v for (k, v) in zip(rep_keys, rep_vals)}
        new_counts = {old2new[k]: v for k, v in counts.iteritems()}
        return new_data, new_counts, vocab, old2new
