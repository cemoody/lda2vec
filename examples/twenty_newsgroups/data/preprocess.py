# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import logging
import pickle

from sklearn.datasets import fetch_20newsgroups
import numpy as np

from lda2vec import preprocess, Corpus

logging.basicConfig()

# Fetch data
remove = ('headers', 'footers', 'quotes')
texts = fetch_20newsgroups(subset='train', remove=remove).data
# Remove tokens with these substrings
bad = set(["ax>", '`@("', '---', '===', '^^^'])


def clean(line):
    return ' '.join(w for w in line.split() if not any(t in w for t in bad))

# Preprocess data
max_length = 10000   # Limit of 10k words per document
# Convert to unicode (spaCy only works with unicode)
texts = [unicode(clean(d)) for d in texts]
tokens, vocab = preprocess.tokenize(texts, max_length, merge=False,
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
# Convert the compactified arrays into bag of words arrays
bow = corpus.compact_to_bow(pruned)
# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)
# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
assert flattened.min() >= 0
# Fill in the pretrained word vectors
n_dim = 300
fn_wordvc = 'GoogleNews-vectors-negative300.bin'
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)
# Save all of the preprocessed files
pickle.dump(vocab, open('vocab.pkl', 'w'))
pickle.dump(corpus, open('corpus.pkl', 'w'))
np.save("flattened", flattened)
np.save("doc_ids", doc_ids)
np.save("pruned", pruned)
np.save("bow", bow)
np.save("vectors", vectors)
