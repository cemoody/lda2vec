import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.cuda import get_array_module


def dirichlet_likelihood(weights, alpha=None):
    """ Calculate the log likelihood of the observed topic proportions.
    A negative likelihood is more likely than a negative likelihood.

    Args:
        weights (chainer.Variable): Unnormalized weight vector. The vector
            will be passed through a softmax function that will map the input
            onto a probability simplex.
        alpha (float): The Dirichlet concentration parameter. Alpha
            greater than 1.0 results in very dense topic weights such
            that each document belongs to many topics. Alpha < 1.0 results
            in sparser topic weights. The default is to set alpha to
            1.0 / n_topics, effectively enforcing the prior belief that a
            document belong to very topics at once.

    Returns:
        ~chainer.Variable: Output loss variable.
    """
    n_documents = weights.data.shape[0]
    n_topics = weights.data.shape[1]
    if alpha is None:
        alpha = 1.0 / n_topics
    if type(weights) is Variable:
        proportions = F.softmax(weights)
    if type(weights) is L.EmbedID:
        np = get_array_module()
        all_docs = Variable(np.arange(n_documents, dtype='int32'))
        proportions = F.softmax(weights(all_docs))
    loss = (alpha - 1.0) * F.log(proportions + 1e-8)
    return -F.sum(loss)
