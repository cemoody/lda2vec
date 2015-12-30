# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it

from lda2vec import preprocess, LDA2Vec
from sklearn.datasets import fetch_20newsgroups

# Fetch data
texts = fetch_20newsgroups(subset='train').data
# Convert to unicode (spaCy only works with unicode)
texts = [unicode(d) for d in texts]

max_length = max(len(doc) for doc in texts)
data, counts, vocab = preprocess.tokenize(texts, max_length, tag=False,
                                          parse=False, entity=False)

# We'll replace very rare words with the pad. The spaCy indices
# roughly correspond to word frequency in English, so we here we drop
# words less frequent than the first 20,000
data[data > 20e3] = -1
counts = {k: v for k, v in counts.items() if k < 20e3}
vocab = {k: v for k, v in vocab.items() if k < 20e3}


# Number of documents
n_docs = len(texts)
# Number of unique words in the vocabulary
n_words = data.max() + 1
# Number of dimensions in a single word vector
n_hidden = 128
# Number of topics to fit
n_topics = 10
# Number of times to pass through the data
epochs = 5

# Initialize the model
model = LDA2Vec(n_words, max_length, n_hidden, counts)
model.add_component(n_docs, n_topics, name='document id')

# Fit the model
for _ in range(epochs):
    model.fit_partial(data, 1.0)
