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
from lda import LDA

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

vocab = pickle.load(open('vocab.pkl', 'r'))
corpus = pickle.load(open('corpus.pkl', 'r'))
bow = np.load("bow.npy").astype('float32')
# Remove bow counts on the first two tokens, which <SKIP> and <EOS>
bow[:, :2] = 0
# Normalize bag of words to be a probability
# bow = bow / bow.sum(axis=1)[:, None]

# Number of docs
n_docs = bow.shape[0]
# Number of unique words in the vocabulary
n_vocab = bow.shape[1]
# Number of dimensions in a single word vector
n_units = 256
# number of topics
n_topics = 20
batchsize = 128
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = LDA(n_docs, n_topics, n_units, n_vocab)
if os.path.exists('lda.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("lda.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)

j = 0
fraction = batchsize * 1.0 / bow.shape[0]
for epoch in range(50000000):
    if epoch % 100 == 0:
        p = cuda.to_cpu(model.proportions.W.data).copy()
        f = cuda.to_cpu(model.factors.W.data).copy()
        w = cuda.to_cpu(model.embedding.W.data).copy()
        d = prepare_topics(p, f, w, words)
        print_top_words_per_topic(d)
    for (ids, batch) in utils.chunks(batchsize, np.arange(bow.shape[0]), bow):
        t0 = time.time()
        optimizer.zero_grads()
        rec, ld = model.forward(ids, batch)
        l = rec + ld
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
        serializers.save_hdf5("lda.hdf5", model)
