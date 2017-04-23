''' Spectral LDA based on tensor CP Decomposition

Example:

    alpha, beta = spectral_lda(docs, alpha0, k)


REFERENCE
-----------
    https://github.com/Mega-DatA-Lab/SpectralLDA-Spark/blob/master/report.pdf

'''
import numpy as np
from rand_svd import rand_svd
from cumulants import moment1, whiten_m3
import mxnet as mx
from proj_l1_simplex import proj_l1_simplex


def spectral_lda(docs, alpha0, k, n_partitions=1):
    ''' Spectral LDA

    Perform Spectral LDA based on tensor CP Decomposition.

    PARAMETERS
    -----------
    docs : n_docs-by-vocab_size array or csr_matrix
        Entire collection of word count vectors.
    alpha0 : float
        Sum of Dirichlet prior parameter.
    k : int
        Rank for the truncated SVD, >= 1.
    n_partitions: int, optional
        Number of partitions, >= 1, 1 by default.

    RETURNS
    -----------
    alpha : length-k array
        Fitted Dirichlet prior parameter for LDA.
    beta : vocab_size-by-k array
        Fitted topic-word distribution for LDA.
    '''
    # pylint: disable=too-many-locals
    def adj_beta(beta):
        ''' Project beta onto l1-simplex

        As the CP Decomposition could return a factor column
        with the inverse sign, project both beta - min(beta) + eps,
        (-beta) - min(-beta) + eps and retain the one with minimal
        shift theta as computed by the Duchi algorithm.
        '''
        # We add 1 / len(beta) to both to make sure the shifts theta
        # are necessarily positive
        beta1 = beta - beta.min() + 1 / len(beta)
        beta2 = (- beta) - (- beta).min() + 1 / len(beta)
        proj_beta, theta1 = proj_l1_simplex(beta1, 1.0)
        proj_neg_beta, theta2 = proj_l1_simplex(beta2, 1.0)

        return proj_beta if theta1 < theta2 else proj_neg_beta

    n_docs, vocab_size = docs.shape
    assert n_docs >= 1 and vocab_size >= 1
    assert alpha0 > 0
    assert k >= 1
    assert n_partitions >= 1

    # Whiten scaled M3 with SVD of scaled M2
    docs_m1 = moment1(docs, n_partitions=n_partitions)
    eigval_m2, eigvec_m2 = rand_svd(docs, alpha0, k, docs_m1=docs_m1,
                                    n_partitions=n_partitions)

    whn = eigvec_m2.dot(np.diag(1.0 / np.sqrt(eigval_m2)))
    whitened_m3 = whiten_m3(docs, whn, alpha0, docs_m1=docs_m1,
                            n_partitions=n_partitions)
    whitened_m3 *= alpha0 * (alpha0 + 1) * (alpha0 + 2) / 2

    # Perform CP Decomposition on whitened M3
    tensor = mx.sym.Variable('t')
    cpdecomp = mx.operator.symbol.CPDecomp3D(tensor, k=k)

    calc = cpdecomp.bind(mx.cpu(),
                         {'t': mx.nd.array(whitened_m3).reshape((k, k, k))})
    cp_results = calc.forward()

    eigval = cp_results[0].asnumpy()
    factors = [mat.asnumpy().T for mat in cp_results[1:]]

    # Recompose alpha, beta
    alpha = 1 / eigval ** 2
    beta = (eigvec_m2.dot(np.diag(np.sqrt(eigval_m2)))
            .dot(factors[0]).dot(np.diag(eigval)))
    for j in range(k):
        beta[:, j] = adj_beta(beta[:, j])

    # Return in descending order of alpha
    return alpha[::-1], beta[:, ::-1]
