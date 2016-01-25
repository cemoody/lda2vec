# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it

from lda2vec import preprocess, LDA2Vec, Corpus
from sklearn.datasets import fetch_20newsgroups
from chainer import serializers
from chainer import cuda
import numpy as np
import os.path
import logging

# Optional: moving the model to the GPU makes it ~10x faster
# set to False if you're having problems with Chainer and CUDA
gpu = cuda.available

logging.basicConfig()

# Fetch data
texts = fetch_20newsgroups(subset='train').data
# Convert to unicode (spaCy only works with unicode)
texts = [unicode(d) for d in texts]

# Preprocess data
max_length = 10000   # Limit of 1k words per document
tokens, vocab = preprocess.tokenize(texts, max_length, tag=False,
                                    parse=False, entity=False)
corpus = Corpus()
# Make a ranked list of rare vs frequent words
corpus.update_word_count(tokens)
corpus.finalize()
# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This builds a new compact index
compact = corpus.to_compact(tokens)
# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=50)
# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)
# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)

# Model Parameters
# Number of documents
n_docs = len(texts)
# Number of unique words in the vocabulary
n_words = flattened.max() + 1
# Number of dimensions in a single word vector
n_hidden = 128
# Number of topics to fit
n_topics = 20
# Get the count for each key
counts = corpus.keys_counts[:n_words]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_words]

# Fit the model
model = LDA2Vec(n_words, n_hidden, counts, dropout_ratio=0.2)
model.add_categorical_feature(n_docs, n_topics, name='document_id')
model.finalize()
if os.path.exists('model.hdf5'):
    serializers.load_hdf5('model.hdf5', model)
for _ in range(200):
    model.top_words_per_topic('document_id', words)
    if gpu:
        model.to_gpu()
    model.fit(flattened, categorical_features=[doc_ids], fraction=1e-3,
              epochs=1)
    serializers.save_hdf5('model.hdf5', model)
    model.to_cpu()
model.top_words_per_topic('document_id', words)

# Visualize the model -- look at lda.ipynb to see the results
model.to_cpu()
topics = model.prepare_topics('document_id', words)
np.savez('topics.pyldavis', **topics)
