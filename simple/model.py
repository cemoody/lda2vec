# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os.path
import logging
import pickle
import random
import time

from sklearn.datasets import fetch_20newsgroups
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import preprocess, Corpus
from simple_lda2vec import SimpleLDA2Vec

logging.basicConfig()

# Fetch data
texts = fetch_20newsgroups(subset='train').data

# Preprocess data
max_length = 10000   # Limit of 10k words per document
if not os.path.exists('doc_ids.npy'):
    # Convert to unicode (spaCy only works with unicode)
    texts = [unicode(d) for d in texts]
    tokens, vocab = preprocess.tokenize(texts, max_length, merge=True,
                                        n_threads=4)
    corpus = Corpus()
    # Make a ranked list of rare vs frequent words
    corpus.update_word_count(tokens)
    corpus.finalize()
    # The tokenization uses spaCy indices, and so may have gaps
    # between indices for words that aren't present in our dataset.
    # This builds a new compact index
    compact = corpus.to_compact(tokens)
    # Remove extremely rare words
    pruned = corpus.filter_count(compact, min_count=30)
    # Words tend to have power law frequency, so selectively
    # downsample the most prevalent words
    clean = corpus.subsample_frequent(pruned)
    # Now flatten a 2D array of document per row and word position
    # per column to a 1D array of words. This will also remove skips
    # and OoV words
    doc_ids = np.arange(pruned.shape[0])
    flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
    # Save all of the preprocessed files
    pickle.dump(vocab, open('vocab.pkl', 'w'))
    pickle.dump(corpus, open('corpus.pkl', 'w'))
    np.save("flattened", flattened)
    np.save("doc_ids", doc_ids)
else:
    vocab = pickle.load(open('vocab.pkl', 'r'))
    corpus = pickle.load(open('corpus.pkl', 'r'))
    flattened = np.load("flattened.npy")
    doc_ids = np.load("doc_ids.npy")

# Optionally, we can initialize our word vectors from a pretrained
# model. This helps when our corpus is small and we'd like to bootstrap
word_vectors = corpus.compact_word_vectors(vocab)

# Model Parameters
# Number of documents
n_docs = len(texts)
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1
# Number of dimensions in a single word vector
# (if using pretrained vectors, should match that dimensionality)
n_units = 256
# Number of topics to fit
n_topics = 20
batchsize = 2048
counts = corpus.keys_counts[:n_vocab]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]
word_vectors = word_vectors[:n_vocab]

model = SimpleLDA2Vec(n_documents=n_docs, n_document_topics=n_topics,
                      n_units=n_units, n_vocab=n_vocab, loss_type='neg_sample',
                      counts=counts)
model.to_gpu()
optimizer = O.Adam()
optimizer.setup(model)


def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    # From stackoverflow question 312443
    keypoints = []
    for i in xrange(0, len(args[0]), n):
        keypoints.append((i, i + n))
    random.shuffle(keypoints)
    for a, b in keypoints:
        yield [arg[a: b] for arg in args]

j = 0
fraction = batchsize * 1.0 / flattened.shape[0]
for epoch in range(100):
    for d, f in chunks(batchsize, doc_ids, flattened):
        t0 = time.time()
        loss = model.fit_pivot(d, f)
        prior = model.prior()
        loss += prior * fraction
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3f} "
               "P:{prior:1.3e} R:{rate:1.3e}")
        prior.to_cpu()
        loss.to_cpu()
        t1 = time.time()
        dt = t1 - t0
        rate = batchsize / dt
        logs = dict(loss=float(loss.data), epoch=epoch, j=j,
                    prior=float(prior.data), rate=rate)
        print msg.format(**logs)
        j += 1
        if j % 100 == 0:
            model.to_cpu()
            for aidx in range(500, 2500, 100):
                sims = model.most_similar(aidx)[::-1]
                ans = vocab.get(corpus.compact_to_loose.get(aidx, None), ' ')
                print ans.strip() + ': ',
                for idx in np.argsort(sims)[::-1][1: 10]:
                    r = vocab.get(corpus.compact_to_loose.get(idx, None), ' ')
                    print (r[:15].strip().replace('\n', '') + " "),
                print ""
            model.to_gpu()
    serializers.save_hdf5("model", model)
