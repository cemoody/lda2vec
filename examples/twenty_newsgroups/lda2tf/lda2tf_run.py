# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it just using co-occurence counts
import os
import os.path
import pickle
import time
import numpy as np
import pandas as pd

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O

from lda2vec import prepare_topics, print_top_words_per_topic
from lda2vec import utils
from lda2tf import LDA2TF

gpu_id = int(os.getenv('CUDA_GPU', 0))
cuda.get_device(gpu_id).use()
print "Using GPU " + str(gpu_id)

data_dir = os.getenv('data_dir', '../data/')
fn_vocab = '{data_dir:s}/vocab.pkl'.format(data_dir=data_dir)
fn_corpus = '{data_dir:s}/corpus.pkl'.format(data_dir=data_dir)
fn_cooc = '{data_dir:s}/coocurrence.pd'.format(data_dir=data_dir)
vocab = pickle.load(open(fn_vocab, 'r'))
corpus = pickle.load(open(fn_corpus, 'r'))
cooc = pd.read_pickle(fn_cooc)

# Model Parameters
# Number of documents
n_docs = cooc.doc_ids.max() + 1
# Number of unique words in the vocabulary
n_vocab = cooc.word_index_x.max() + 1
# 'Strength' of the dircihlet prior; 200.0 seems to work well
clambda = 200.0
# Number of topics to fit
n_topics = int(os.getenv('n_topics', 20))
batchsize = 4096 * 8
# Number of dimensions in a single word vector
n_units = int(os.getenv('n_units', 300))
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]
# How many tokens are in each document
vc = cooc.groupby('doc_ids')['counts'].sum()
doc_lengths = np.zeros(cooc.doc_ids.values.max() + 1, dtype='int32')
doc_lengths[vc.index] = vc.values
# Count all token frequencies
vc = cooc.groupby('word_index_x')['counts'].sum()
term_frequency = np.zeros(cooc.word_index_x.values.max() + 1, dtype='int32')
term_frequency[vc.index] = vc.values

model = LDA2TF(n_vocab, n_docs, n_topics, n_units, k=15.0)
if os.path.exists('lda2tf.hdf5'):
    print "Reloading from saved"
    serializers.load_hdf5("lda2tf.hdf5", model)
# model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)
clip = chainer.optimizer.GradientClipping(5.0)
optimizer.add_hook(clip)

j = 0
epoch = 0
fraction = batchsize * 1.0 / cooc.shape[0]
for epoch in range(5000):
    data = prepare_topics(cuda.to_cpu(model.mixture.weights.W.data).copy(),
                          cuda.to_cpu(model.mixture.factors.W.data).copy(),
                          cuda.to_cpu(model.embed.W.data).copy(),
                          words)
    print_top_words_per_topic(data)
    data['doc_lengths'] = doc_lengths
    data['term_frequency'] = term_frequency
    np.savez('topics.pyldavis', **data)
    cnt_obs = cooc['counts'].sum() * 1.0
    # Shuffle dataset
    cooc.reindex(np.random.permutation(cooc.index))
    for (sample,) in utils.chunks(batchsize, cooc):
        t0 = time.time()
        optimizer.zero_grads()
        idxx = sample['word_index_x'].values
        idxy = sample['word_index_y'].values
        idxd = sample['doc_ids'].values
        cntx = sample['cnt_word_index_x'].values * 1.0
        cnty = sample['cnt_word_index_y'].values * 1.0
        cntd = sample['cnt_doc_ids'].values * 1.0
        cnt_joint = sample['counts'].values * 1.0
        llh = model.forward(idxx, idxy, idxd, cnt_joint, cntx, cnty,
                            cntd, cnt_obs)
        prior = model.prior()
        loss = llh + prior * fraction
        loss.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{llh:1.3e} "
               "P:{prior:1.3e} R:{rate:1.3e}")
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(llh=float(llh.data), epoch=epoch, j=j,
                    prior=float(prior.data), rate=rate)
        print msg.format(**logs)
        j += 1
    serializers.save_hdf5("lda2tf.hdf5", model)
