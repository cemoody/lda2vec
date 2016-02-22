# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This example loads a large 800MB Hacker News comments dataset
# and trains a multi-component lda2vec model on it

from lda2vec import preprocess, Corpus
import numpy as np
import pandas as pd
import logging
import pickle
import os.path

logging.basicConfig()

max_length = 250   # Limit of 250 words per comment
min_author_comments = 10  # Exclude authors with fewer comments
min_story_comments = 10  # Exclude stories with fewer comments
nrows = None  # Number of rows of file to read; None reads in full file

fn = "hacker_news_comments.csv"
url = "https://zenodo.org/record/45901/files/hacker_news_comments.csv"
if not os.path.exists(fn):
    import requests
    response = requests.get(url, stream=True, timeout=2400)
    with open(fn, 'w') as fh:
        # Iterate over 1MB chunks
        for data in response.iter_content(1024**2):
            fh.write(data)


features = []
# Convert to unicode (spaCy only works with unicode)
features = pd.read_csv(fn, encoding='utf8', nrows=nrows)
# Convert all integer arrays to int32
for col, dtype in zip(features.columns, features.dtypes):
    if dtype is np.dtype('int64'):
        features[col] = features[col].astype('int32')

# Tokenize the texts
# If this fails try running python -m spacy.en.download all --force
texts = features.pop('comment_text').values
tokens, vocab = preprocess.tokenize(texts, max_length, tag=False, n_threads=4,
                                    parse=False, entity=False, merge=True)
del texts

# Make a ranked list of rare vs frequent words
corpus = Corpus()
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

# Extract numpy arrays over the fields we want covered by topics
# Convert to categorical variables
author_id = pd.Categorical(features['comment_author']).codes
story_id = pd.Categorical(features['story_id']).codes
# Chop dates into 100 epochs
time_id = pd.cut(pd.Categorical(features['story_time']), 100).codes

# Extract outcome supervised features
ranking = features['comment_ranking'].values
score = features['story_comment_count'].values

# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
feature_arrs = (story_id, author_id, time_id, ranking, score)
flattened, features_flat = corpus.compact_to_flat(pruned, *feature_arrs)
# Flattened feature arrays
(story_id_f, author_id_f, time_id_f, ranking_f, score_f) = features_flat

# Save the data
pickle.dump(corpus, open('corpus', 'w'))
pickle.dump(vocab, open('vocab', 'w'))
features.to_pickle('features.pd')
data = dict(flattened=flattened, story_id=story_id_f, author_id=author_id_f,
            time_id=time_id_f, ranking=ranking_f, score=score_f)
np.savez('data', data)
np.save(open('tokens', 'w'), tokens)
