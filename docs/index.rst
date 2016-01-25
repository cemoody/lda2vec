==============================================
lda2vec -- flexible & interpretable NLP models
==============================================

This is the documentation for lda2vec, a framework for useful 
flexible and interpretable NLP models.

Defining the model is simple and quick::

    model = LDA2Vec(n_words, max_length, n_hidden, counts)
    model.add_component(n_docs, n_topics, name='document id')
    model.fit(clean, components=[doc_ids])

While visualizing the feature is similarly straightforward::

    topics = model.prepare_topics('document_id', vocab)
    prepared = pyLDAvis.prepare(topics)
    pyLDAvis.display(prepared)

Resources
---------
See this `Jupyter Notebook <http://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty_newsgroups/lda.ipynb>`_
for an example of an end-to-end demonstration.

See this `presentation <http://www.slideshare.net/ChristopherMoody3/word2vec-lda-and-introducing-a-new-hybrid-algorithm-lda2vec-57135994>`_
for a presentation focused on the benefits of word2vec, LDA, and lda2vec.

See the `API reference docs <https://lda2vec.readthedocs.org/en/latest/>`_

See the `GitHub repo <https://github.com/cemoody/lda2vec>`_

API
===
.. toctree::

    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
