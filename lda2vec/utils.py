from chainer import Variable
import random
import numpy as np


def move(xp, *args):
    for arg in args:
        if 'float' in str(arg.dtype):
            yield Variable(xp.asarray(arg, dtype='float32'))
        else:
            assert 'int' in str(arg.dtype)
            yield Variable(xp.asarray(arg, dtype='int32'))


def most_similar(embeddings, word_index):
    input_vector = embeddings.W[word_index]
    similarities = embeddings.dot(input_vector)
    return similarities


def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    keypoints = []
    for i in xrange(0, len(args[0]), n):
        keypoints.append((i, i + n))
    random.shuffle(keypoints)
    for a, b in keypoints:
        yield [arg[a: b] for arg in args]


class MovingAverage():
    def __init__(self, lastn=100):
        self.points = np.array([])
        self.lastn = lastn

    def add(self, x):
        self.points = np.append(self.points, x)

    def mean(self):
        return np.mean(self.points[-self.lastn:])

    def std(self):
        return np.std(self.points[-self.lastn:])

    def get_stats(self):
        return (np.mean(self.points[-self.lastn:]),
                np.std(self.points[-self.lastn:]))
