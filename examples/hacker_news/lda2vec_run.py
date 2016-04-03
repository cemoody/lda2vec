# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import pickle
import time

from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic
from lda2vec_model import LDA2Vec

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

# You must run preprocess.py before this data becomes available
vocab = pickle.load(open('vocab', 'r'))
corpus = pickle.load(open('corpus', 'r'))
data = np.load(open('data.npz', 'r'))
flattened = data['flattened']
story_id = data['story_id']
author_id = data['author_id']
time_id = data['time_id']
ranking = data['ranking'].astype('float32')
score = data['score'].astype('float32')

# Model Parameters
# Number of documents
n_stories = story_id.max() + 1
# Number of users
n_authors = author_id.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# Number of dimensions in a single word vector
n_units = 256
# 'Strength' of the dircihlet prior; 200.0 seems to work well
clambda = 200.0
# Number of topics to fit
n_story_topics = 40
n_author_topics = 20
batchsize = 4096 * 2
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = LDA2Vec(n_stories, n_story_topics, n_authors, n_author_topics,
                n_units=n_units, n_vocab=n_vocab, counts=counts,
                n_samples=7)
if os.path.exists('lda2vec_hn.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("lda2vec_hn.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)

j = 0
epoch = 0
fraction = batchsize * 1.0 / flattened.shape[0]
for epoch in range(5000):
    print "Story topics"
    w = cuda.to_cpu(model.mixture_stories.weights.W.data).copy()
    f = cuda.to_cpu(model.mixture_stories.factors.W.data).copy()
    v = cuda.to_cpu(model.embed.W.data).copy()
    d = prepare_topics(w, f, v, words)
    print_top_words_per_topic(d)
    print "Author topics"
    w = cuda.to_cpu(model.mixture_authors.weights.W.data).copy()
    f = cuda.to_cpu(model.mixture_authors.factors.W.data).copy()
    d = prepare_topics(w, f, v, words)
    print_top_words_per_topic(d)
    for s, a, f in utils.chunks(batchsize, story_id, author_id, flattened):
        t0 = time.time()
        l = model.fit_partial(s.copy(), a.copy(), f.copy())
        prior = model.prior()
        loss = l + prior * fraction * clambda
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
               "P:{prior:1.3e} R:{rate:1.3e}")
        prior.to_cpu()
        loss.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(loss=float(l.data), epoch=epoch, j=j,
                    prior=float(prior.data), rate=rate)
        print msg.format(**logs)
        j += 1
    serializers.save_hdf5("lda2vec_hn.hdf5", model)
