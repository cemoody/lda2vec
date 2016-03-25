# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import pickle
import time
import random

from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from nvdm import NVDM

vocab = pickle.load(open('vocab.pkl', 'r'))
corpus = pickle.load(open('corpus.pkl', 'r'))
pruned = np.load("pruned.npy")
bow = np.load("bow.npy")
# Set the <SKIP> tokens to -1, which will be ignored in the loss function
pruned[pruned == 0] = -1
# Remove bow counts on the first two tokens, which <SKIP> and <EOS>
bow[:, :2] = 0

# Split up documents into chunks
chunksize = 10000
batchsize = 2
chunks = []
for b, p in utils.chunks(batchsize, bow, pruned[:, :9000]):
    for chunk, in utils.chunks(chunksize, p.T):
        if np.all(chunk == -1):
            continue
        flattened = chunk.T.flatten()
        chunks.append((b, flattened))

# Number of unique words in the vocabulary
n_vocab = pruned.max() + 1
# Number of dimensions in a single word vector
n_units = 256
n_topics = 20
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = NVDM(n_vocab, n_topics, n_units)
if os.path.exists('nvdm.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("nvdm.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)

j = 0
fraction = batchsize * 1.0 / pruned.shape[0]
for epoch in range(500):
    random.shuffle(chunks)
    for b, flattened in chunks:
        t0 = time.time()
        rec, kl, dl = model.fit(b, flattened)
        optimizer.zero_grads()
        l = rec + kl + dl
        l.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
               "P:{prior:1.3e} DL:{dl:1.3e} R:{rate:1.3e}")
        l.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(loss=float(rec.data), epoch=epoch, j=j,
                    prior=float(kl.data), rate=rate, dl=float(dl.data))
        print msg.format(**logs)
        j += 1
    serializers.save_hdf5("nvdm.hdf5", model)
