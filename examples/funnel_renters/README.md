
This directory explores using lda2vec to train a word and topic model for messages from prospective renters.

# Run

Navigate into the `data` directory, then run `download_google_news_vectors.sh` to download the Google News word vectors. Then run:

```bash
python preprocess.py
```

The script assumes you have a CSV file of messages called `messages_100k.csv` to preprocess. When the script finishes, you should have the following files: `vocab.pkl`, `corpus.pkl`, `bow.npy`, `doc_ids.npy`, `flattened.npy`, `pruned.npy`, and `vectors.npy`.

Next, move into the `lda2vec` directory, then run:

```bash
python lda2vec_run.py
```

It should save the files: `topics.pyldavis.npz`, `lda2vec.hdf5`, and `progress.shelve.dat`.

You can then visualize the results with `lda2vec.ipynb`.
