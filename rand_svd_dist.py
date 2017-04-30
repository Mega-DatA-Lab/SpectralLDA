''' Randomised SVD for scaled M2 in distributed mode

'''
import numpy as np
import scipy.linalg
from cumulants_dist import moment1_dist, prod_m2_x_dist

def rand_svd_dist(docs, alpha0, k, docs_m1=None, n_iter=1, n_partitions=1):
    ''' Randomised SVD in distributed mode

    Perform Randomised SVD on scaled M2.

    PARAMETERS
    -----------
    docs : str
        Path for the entire collection of word count vectors.
    alpha0 : float
        Sum of Dirichlet prior parameter.
    k : int
        Rank for the truncated SVD, >= 1.
    docs_m1: length-vocab_size array, optional
        M1 of the entire collection of word count vectors.
    n_iter: int, optional
        Number of iterations for the Krylov method, >= 0, 1 by default.
    n_partitions: int, optional
        Number of partitions, >= 1, 1 by default.

    RETURNS
    -----------
    eigval : length-k array
        Top k eigenvalues of scaled M2.
    eigvec : vocab_size-by-k array
        Top k eigenvectors of scaled M2.
    '''
    # pylint: disable=too-many-arguments
    n_docs, vocab_size = docs.shape
    assert n_docs >= 1 and vocab_size >= 1
    if docs_m1 is not None:
        assert docs_m1.ndim == 1 and vocab_size == docs_m1.shape[0]
    assert alpha0 > 0
    assert k >= 1
    assert n_iter >= 0
    assert n_partitions >= 1

    # Augment slightly k for better convergence
    k_aug = np.min([k + 5, vocab_size])

    # Gaussian test matrix
    test_x = np.random.randn(vocab_size, k_aug)

    # Krylov method
    if docs_m1 is None:
        docs_m1 = moment1_dist(docs, n_partitions=n_partitions)
    for _ in range(2 * n_iter + 1):
        prod_test = prod_m2_x_dist(docs, test_x, alpha0,
                                   docs_m1=docs_m1, n_partitions=n_partitions)
        test_x, _ = scipy.linalg.qr(prod_test, mode='economic')

    # X^T M2 M2 X = Q S Q^T
    # If M2 M2 = X Q S Q^T X^T, then the above holds,
    # where X is an orthonormal test matrix.
    prod_test = prod_m2_x_dist(docs, test_x, alpha0,
                               n_partitions=n_partitions)
    prod_test *= alpha0 * (alpha0 + 1)
    svd_q, svd_s, _ = scipy.linalg.svd(prod_test.T.dot(prod_test))

    return np.sqrt(svd_s)[:k], test_x.dot(svd_q)[:, :k]
