# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import pickle
import time

from chainer import serializers
from chainer import cuda
import chainer.optimizers as O
import numpy as np

from lda2vec import prepare_topics, print_top_words_per_topic
from lda2vec import utils
from nslda import NSLDA

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

vocab = pickle.load(open('../data/vocab.pkl', 'r'))
corpus = pickle.load(open('../data/corpus.pkl', 'r'))
doc_id = np.load("../data/doc_ids.npy")
flattened = np.load("../data/flattened.npy")

# Number of docs
n_docs = doc_id.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# Number of dimensions in a single word vector
n_units = 256
# number of topics
n_topics = 20
batchsize = 4096 * 8
# Strength of Dirichlet prior
strength = 1.0
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = NSLDA(counts, n_docs, n_topics, n_units, n_vocab)
if os.path.exists('nslda.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("nslda.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)

j = 0
fraction = batchsize * 1.0 / flattened.shape[0]
for epoch in range(50000000):
    p = cuda.to_cpu(model.proportions.W.data).copy()
    f = cuda.to_cpu(model.factors.W.data).copy()
    w = cuda.to_cpu(model.loss_func.W.data).copy()
    d = prepare_topics(p, f, w, words)
    print_top_words_per_topic(d)
    for (doc_ids, flat) in utils.chunks(batchsize, doc_id, flattened):
        t0 = time.time()
        optimizer.zero_grads()
        rec, ld = model.forward(doc_ids, flat)
        l = rec + ld * fraction * strength
        l.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{rec:1.3e} "
               "P:{ld:1.3e} R:{rate:1.3e}")
        l.to_cpu()
        rec.to_cpu()
        ld.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(rec=float(rec.data), epoch=epoch, j=j,
                    ld=float(ld.data), rate=rate)
        print msg.format(**logs)
        j += 1
    if epoch % 100 == 0:
        serializers.save_hdf5("nslda.hdf5", model)
