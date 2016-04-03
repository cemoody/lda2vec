# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import pickle
import time

from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from nvdm import NVDM

vocab = pickle.load(open('vocab.pkl', 'r'))
corpus = pickle.load(open('corpus.pkl', 'r'))
bow = np.load("bow.npy").astype('float32')
# Remove bow counts on the first two tokens, which <SKIP> and <EOS>
bow[:, :2] = 0
# Normalize bag of words to be a probability
bow = bow / bow.sum(axis=1)[:, None]

# Number of unique words in the vocabulary
n_vocab = bow.shape[1]
# Number of dimensions in a single word vector
n_units = 256
batchsize = 128
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = NVDM(n_vocab, n_units)
if os.path.exists('nvdm.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("nvdm.hdf5", model)
# model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)

j = 0
fraction = batchsize * 1.0 / bow.shape[0]
for epoch in range(500):
    for (batch,) in utils.chunks(batchsize, bow):
        t0 = time.time()
        rec, kl = model.observe(batch)
        optimizer.zero_grads()
        l = rec + kl
        l.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{rec:1.3e} "
               "P:{kl:1.3e} R:{rate:1.3e}")
        l.to_cpu()
        rec.to_cpu()
        kl.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(rec=float(rec.data), epoch=epoch, j=j,
                    kl=float(kl.data), rate=rate)
        print msg.format(**logs)
        j += 1
    serializers.save_hdf5("nvdm.hdf5", model)
