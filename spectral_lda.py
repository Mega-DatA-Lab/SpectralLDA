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
    def factor_correct_sign(factors, eps=1e-6):
        ''' Return unique factor matrix with correct signs '''
        _, k = factors[0].shape
        factor = np.zeros_like(factors[0])

        for j in range(k):
            if np.linalg.norm(factors[1][:, j] - factors[2][:, j]) < eps:
                factor[:, j] = factors[0][:, j]
            elif np.linalg.norm(factors[0][:, j] - factors[2][:, j]) < eps:
                factor[:, j] = factors[1][:, j]
            elif np.linalg.norm(factors[0][:, j] - factors[1][:, j]) < eps:
                factor[:, j] = factors[2][:, j]
            else:
                raise RuntimeError('Invalid results from CPDecomp.')

        return factor

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

    # Unique factor matrix correcting signs of columns
    # in all factor matrices
    unique_factor = factor_correct_sign(factors)

    # Recompose alpha, beta
    alpha = 1 / eigval ** 2
    beta = (eigvec_m2.dot(np.diag(np.sqrt(eigval_m2)))
            .dot(unique_factor).dot(np.diag(eigval)))
    for j in range(k):
        beta[:, j] = proj_l1_simplex(beta[:, j], 1.0)

    # Return in descending order of alpha
    return alpha[::-1], beta[:, ::-1]
