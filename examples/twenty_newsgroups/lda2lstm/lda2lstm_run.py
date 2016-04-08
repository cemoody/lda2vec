# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import pickle
import time

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2lstm import LDA2LSTM

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

vocab = pickle.load(open('../data/vocab.pkl', 'r'))
corpus = pickle.load(open('../data/corpus.pkl', 'r'))
compact = np.load("../data/compact.npy")
doc_ids = np.load("../data/doc_ids.npy")

# Model Parameters
# Number of documents
n_docs = doc_ids.max() + 1
# Number of unique words in the vocabulary
n_vocab = compact.max() + 1
# Number of dimensions in a single word vector
n_units = 256
# Number of topics to fit
n_topics = 20
batchsize = 256
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = LDA2LSTM(n_documents=n_docs, n_document_topics=n_topics,
                 n_units=n_units, n_vocab=n_vocab)
if os.path.exists('lda2lstm.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("lda2vec.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)
clip = chainer.optimizer.GradientClipping(5.0)
optimizer.add_hook(clip)

j = 0
epoch = 0
fraction = batchsize * 1.0 / compact.shape[0]
for epoch in range(5000):
    for d, f in utils.chunks(batchsize, doc_ids, compact):
        t0 = time.time()
        optimizer.zero_grads()
        l, kl = model.fit_partial(d.copy(), f.copy())
        prior = model.prior()
        loss = l + kl + prior * fraction
        loss.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
               "P:{prior:1.3e} KL:{kl:1.3e} R:{rate:1.3e}")
        prior.to_cpu()
        loss.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(loss=float(l), epoch=epoch, j=j, kl=float(kl.data),
                    prior=float(prior.data), rate=rate)
        print msg.format(**logs)
        j += 1
    serializers.save_hdf5("lda2lstm.hdf5", model)
