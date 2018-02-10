''' Spectral LDA based on tensor CP Decomposition

Example:

    alpha, beta = spectral_lda(docs, alpha0, k)


REFERENCE
-----------
    https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet/blob/master/report.pdf

'''
import sys
import numpy as np
from rand_svd import rand_svd
from cumulants import moment1, whiten_m3
import tensorly
from tensorly.decomposition import parafac
from proj_l1_simplex import proj_l1_simplex

tensorly.set_backend('numpy')

def spectral_lda(docs, alpha0, k, l1_simplex_proj=False, n_partitions=1):
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
    l1_simplex_proj : bool, optional
        Projection of topic-word-distribution into the l1-simplex, False by default.
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
    def factor_correct_sign(factors):
        ''' Return unique factor matrix with correct signs '''
        _, k = factors[0].shape
        factor = np.zeros_like(factors[0])

        for j in range(k):
            diff_ = [np.linalg.norm(factors[1][:, j] - factors[2][:, j]),
                     np.linalg.norm(factors[0][:, j] - factors[2][:, j]),
                     np.linalg.norm(factors[0][:, j] - factors[1][:, j])]
            factor[:, j] = factors[np.argmin(diff_)][:, j]

        return factor

    n_docs, vocab_size = docs.shape
    assert n_docs >= 1 and vocab_size >= 1
    assert alpha0 > 0
    assert k >= 1
    assert n_partitions >= 1

    _valid_docs = np.squeeze(np.array(docs.sum(axis=1))) >= 3
    print(f'# docs: {docs.shape[0]}\t# valid: {_valid_docs.sum()}',
          file=sys.stderr)
    docs = docs[_valid_docs]

    # Whiten scaled M3 with SVD of scaled M2
    docs_m1 = moment1(docs, n_partitions=n_partitions)
    eigval_m2, eigvec_m2 = rand_svd(docs, alpha0, k, docs_m1=docs_m1,
                                    n_iter=3, n_partitions=n_partitions)

    whn = eigvec_m2.dot(np.diag(1.0 / np.sqrt(eigval_m2)))
    whitened_m3 = whiten_m3(docs, whn, alpha0, docs_m1=docs_m1,
                            n_partitions=n_partitions)
    whitened_m3 *= alpha0 * (alpha0 + 1) * (alpha0 + 2) / 2
    # Fold into order-3 tensor
    whitened_m3 = whitened_m3.reshape((k, k, k))

    # Perform CP Decomposition on whitened M3
    factors = parafac(whitened_m3, k)
    eigval = np.linalg.norm(factors[0], axis=0)
    factors[0] /= eigval
    assert np.allclose(np.linalg.norm(factors[0], axis=0), 1)

    # Unique factor matrix correcting signs of columns
    # in all factor matrices
    unique_factor = factor_correct_sign(factors)

    # Recompose alpha, beta
    alpha = 1 / eigval ** 2
    beta = (eigvec_m2.dot(np.diag(np.sqrt(eigval_m2)))
            .dot(unique_factor).dot(np.diag(eigval)))
    if l1_simplex_proj:
        for j in range(k):
            beta[:, j] = proj_l1_simplex(beta[:, j], 1.0)

    # Return in descending order of alpha
    return alpha[::-1], beta[:, ::-1]
