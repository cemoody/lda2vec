# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it

from lda2vec import LDA2Vec
from chainer import serializers
from chainer import cuda
import pandas as pd
import numpy as np
import os.path
import logging
import cPickle as pickle

# Optional: moving the model to the GPU makes it ~10x faster
# set to False if you're having problems with Chainer and CUDA
gpu = cuda.available

logging.basicConfig()

# You must run preprocess.py before this data becomes available
# Load the data
corpus = pickle.load(open('corpus', 'r'))
vocab = pickle.load(open('vocab', 'r'))
features = pd.read_pickle('features.pd')
data = np.load(open('data.npz', 'r'))
flattened = data['flattened']
story_id = data['story_id']
author_id = data['author_id']
time_id = data['time_id']
ranking = data['ranking'].astype('float32')
score = data['score'].astype('float32')

# Model Parameters
# Number of documents
n_stories = story_id.max() + 1
# Number of authors
n_authors = author_id.max() + 1
# Number of time periods
n_times = time_id.max() + 1
# Number of unique words in the vocabulary
n_words = flattened.max() + 1
# Number of dimensions in a single word vector
# (if using pretrained vectors, should match that dimensionality)
n_hidden = 300
# Number of topics to fit for types of stories
n_topic_stories = 50
# Number of topics to fit for types of authors
n_topic_authors = 20
# Number of topics to fit for types of days
n_topic_times = 10
# Get the count for each key
counts = corpus.keys_counts[:n_words]
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_words]

print "n_words", n_words
print "n_stories", n_stories
print "n_authors", n_authors
print "n_times", n_times

# Fit the model
model = LDA2Vec(n_words, n_hidden, counts, dropout_ratio=0.5, n_samples=5)
# We want topics over different articles, but we want those topics
# to correlate with the article 'score'. This gives us a better idea
# of what topics get to the top of HN
model.add_categorical_feature(n_stories, n_topic_stories, name='story_id',
                              loss_type='mean_squared_error', n_target_out=1,
                              logdet_penalty=1e-3)
# This gives us topics over comments on HN. This lets us figure out
# what categories of comments get ranked higher than others.
# Note that we're assuming the loss function is still MSE,
# even though the rank isn't really normally distributed.
model.add_categorical_feature(n_authors, n_topic_authors, name='author_id')
# We have topics over different parts in the evolution of Hacker News
# but we won't have any outcome variables for it, so don't define
# a loss_type.
# model.add_categorical_feature(n_times, n_topic_times, name='time_id',
#                               logdet_penalty=1e-3)
model.finalize()

# Reload model if pre-existing
if os.path.exists('model.hdf5'):
    print "Reloading from existing"
    serializers.load_hdf5('model.hdf5', model)

# Train the model
cat_feats = [story_id, author_id]
targets = [score]
for _ in range(200):
    model.to_cpu()
    model.top_words_per_topic('story_id', words)
    model.top_words_per_topic('author_id', words)
    if gpu:
        model.to_gpu()
    model.fit(flattened, categorical_features=cat_feats, fraction=16e-5,
              epochs=1, targets=targets)
    serializers.save_hdf5('model.hdf5', model)

model.to_cpu()
model.top_words_per_topic('story_id', words)
model.top_words_per_topic('author_id', words)

# Visualize the model -- look at model.ipynb to see the results
for component in ['story_id', 'author_id', 'time_id']:
    topics = model.prepare_topics(component, words)
    np.savez('topics.%s.pyldavis' % component, **topics)
