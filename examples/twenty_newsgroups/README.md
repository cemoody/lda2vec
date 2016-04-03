There are a variety of LDA-inspired models in this directory. 

Warning: Not all of these models are working at the moment.


lda
----
This model takes a document id, finds the latent document vector, and predicts a bag-of-words (BoW) representation for each document.


nslda
-----
Like the above, but predicts individual words instead of a BoW representation predicts individual words using word2vec's negative sampling. Note that this is not using skipgrams; it simply maps context vector to word vector while also discouraging the mapping from context vector to negatively sampled words.


lda2vec
-------
This model adds in skipgrams. A word predicts another word in the same window, as in word2vec, but also has the notion of a context vector which only changes at the document level as in LDA. 

nvdm
----

This code implements the Neural Inference Document Modeling (NVDM) in the paper ["Neural Variational Inference for Text Processing".](http://arxiv.org/pdf/1511.06038v3.pdf)