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

API
===
.. toctree::

    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
