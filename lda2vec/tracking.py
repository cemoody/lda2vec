import numpy as np
from sklearn.linear_model import LinearRegression


class Tracking:
    cache = {}
    calls = 0
    slope = 0.0

    def __init__(self, n=5000):
        """ The tracking class keeps a most recently used cache of values

        Parameters
        ----------
        n: int
        Number of items to keep.
        """
        self.n = n

    def add(self, key, item):
        """ Add an item with a particular to the cache.

        >>> tracker = Tracking()
        >>> tracker.add('log_perplexity', 55.6)
        >>> tracker.cache['log_perplexity']
        [55.6]
        >>> tracker.add('log_perplexity', 55.2)
        >>> tracker.add('loss', -12.1)
        >>> tracker.cache['log_perplexity']
        [55.6, 55.2]
        >>> tracker.cache['loss']
        [-12.1]
        """
        if key not in self.cache:
            self.cache[key] = []
        self.cache[key].append(item)
        if len(self.cache[key]) > self.n:
            self.cache[key] = self.cache[key][:self.n]

    def stats(self, key):
        """ Get the statistics for items with a particular key

        >>> tracker = Tracking()
        >>> tracker.add('log_perplexity', 55.6)
        >>> tracker.add('log_perplexity', 55.2)
        >>> tracker.stats('log_perplexity')
        (55.400000000000006, 0.19999999999999929, 0.0)
        """
        data = self.cache[key]
        mean = np.mean(data)
        std = np.std(data)
        slope = self.slope
        if self.calls % 100 == 0:
            lr = LinearRegression()
            x = np.arange(len(data)).astype('float32')
            lr.fit(x[:, None], np.array(data))
            self.slope = lr.coef_[0]
        self.calls += 1
        return mean, std, slope

if __name__ == "__main__":
    import doctest
    doctest.testmod()
