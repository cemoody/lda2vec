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
# from lda2vec import prepare_topics, print_top_words_per_topic
from lda2vec_model import LDA2Vec

from chainer import cuda
gpu_id = int(os.getenv('CUDA_GPU'))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

vocab = pickle.load(open('vocab.pkl', 'r'))
corpus = pickle.load(open('corpus.pkl', 'r'))
flattened = np.load("flattened.npy")
doc_ids = np.load("doc_ids.npy")

# Model Parameters
# Number of documents
n_docs = doc_ids.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# Number of dimensions in a single word vector
n_units = 256
# 'Strength' of the dircihlet prior; 200.0 seems to work well
clambda = 200.0
# Number of topics to fit
n_topics = 32
batchsize = 4096 + 2048
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]

model = LDA2Vec(n_documents=n_docs, n_document_topics=n_topics,
                n_units=n_units, n_vocab=n_vocab, counts=counts,
                n_samples=15)
if os.path.exists('lda2vec.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("lda2vec.hdf5", model)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)

j = 0
epoch = 0
fraction = batchsize * 1.0 / flattened.shape[0]
while True:
    epoch += 1
    model.to_cpu()
    # data = prepare_topics(model.mixture.weights.W.data.copy(),
    #                       model.mixture.factors.W.data.copy(),
    #                       model.embed.W.data.copy(),
    #                       words)
    # print_top_words_per_topic(data)
    model.to_gpu()
    for d, f in utils.chunks(batchsize, doc_ids, flattened):
        t0 = time.time()
        l = model.fit_partial(d, f)
        prior = model.prior()
        loss = l + prior * fraction
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
    serializers.save_hdf5("lda2vec.hdf5", model)
