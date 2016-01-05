# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it

from lda2vec import preprocess, LDA2Vec, Corpus
from sklearn.datasets import fetch_20newsgroups
from chainer import serializers
import numpy as np
import logging

logging.basicConfig()

# Fetch data
texts = fetch_20newsgroups(subset='train').data
# Convert to unicode (spaCy only works with unicode)
texts = [unicode(d) for d in texts]

# Preprocess data
max_length = 10000   # Limit of 10k words per document
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
pruned = corpus.filter_count(compact, min_count=5)
# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)
# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(clean.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(clean, doc_ids)
# Get the count for each key
counts = corpus.keys_counts

# Model Parameters
# Number of documents
n_docs = len(texts)
# Number of unique words in the vocabulary
n_words = flattened.max() + 1
# Number of dimensions in a single word vector
n_hidden = 128
# Number of topics to fit
n_topics = 20

# Fit the model
model = LDA2Vec(n_words, n_hidden, counts)
model.add_categorical_feature(n_docs, n_topics, name='document id')
model.finalize()
# Move the model to the GPU makes it ~50x faster
model.to_gpu()
model.fit(flattened, categorical_features=[doc_ids], fraction=1e-3, epochs=50)
serializers.save_hdf5('model.hdf5', model)
