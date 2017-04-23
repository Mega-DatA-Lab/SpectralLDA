''' Randomised SVD for scaled M2

'''
import os
import subprocess
import numpy as np
import scipy.linalg
from cumulants import moment1, prod_m2_x
from utils import sync_dir

def rand_svd(docs, alpha0, k, docs_m1=None, n_iter=1, n_partitions=1):
    ''' Randomised SVD in local mode

    Perform Randomised SVD on scaled M2.

    PARAMETERS
    -----------
    docs : n_docs-by-vocab_size array or csr_matrix
        Entire collection of word count vectors.
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
    k_aug = np.min([k + 4, vocab_size, 2 * k])

    # Gaussian test matrix
    test_x = np.random.randn(vocab_size, k_aug)

    # Krylov method
    if docs_m1 is None:
        docs_m1 = moment1(docs, n_partitions=n_partitions)
    for _ in range(2 * n_iter + 1):
        prod_test = prod_m2_x(docs, test_x, alpha0,
                              docs_m1=docs_m1, n_partitions=n_partitions)
        test_x, _ = scipy.linalg.qr(prod_test, mode='economic')

    # X^T M2 M2 X = Q S Q^T
    # If M2 M2 = X Q S Q^T X^T, then the above holds,
    # where X is an orthonormal test matrix.
    prod_test = prod_m2_x(docs, test_x, alpha0,
                          n_partitions=n_partitions)
    prod_test *= alpha0 * (alpha0 + 1)
    svd_q, svd_s, _ = scipy.linalg.svd(prod_test.T.dot(prod_test))

    return np.sqrt(svd_s)[:k], test_x.dot(svd_q)[:, :k]


def rand_svd_dist(docs_file, params_file, q):
    # pylint: disable

    def launch_compute_M1_dist():
        cmd = '{} --launcher ssh'.format(os.environ['PS_LAUNCH'])
        cmd += ' -n {}'.format(os.environ['DMLC_NUM_WORKER'])
        cmd += ' -s {}'.format(os.environ['DMLC_NUM_SERVER'])
        cmd += ' -H {}'.format(os.environ['DMLC_HOSTFILE'])
        cmd += ' --sync-dst-dir {}'.format(os.environ['LDA_DATA_PATH'])
        cmd += ' python3 cumulants_dist.py compute_M1'
        cmd += ' --data-path {}'.format(os.environ['LDA_DATA_PATH'])

        subprocess.run([cmd])
        sync_dir(os.environ['LDA_DATA_PATH'], dmlc_workers[0],
                 os.environ['LDA_DATA_PATH'])

        return np.load(params_file)['M1']

    def launch_mult_M2_X_dist(X):
        np.savez(params_file, X=X, M1=M1, alpha0=alpha0, k=k)

        cmd = '{} --launcher ssh'.format(os.environ['PS_LAUNCH'])
        cmd += ' -n {}'.format(os.environ['DMLC_NUM_WORKER'])
        cmd += ' -s {}'.format(os.environ['DMLC_NUM_SERVER'])
        cmd += ' -H {}'.format(os.environ['DMLC_HOSTFILE'])
        cmd += ' --sync-dst-dir {}'.format(os.environ['LDA_DATA_PATH'])
        cmd += ' python3 cumulants_dist.py mult_M2_X'
        cmd += ' --data-path {}'.format(os.environ['LDA_DATA_PATH'])

        subprocess.run([cmd])
        sync_dir(os.environ['LDA_DATA_PATH'], dmlc_workers[0],
                 os.environ['LDA_DATA_PATH'])

        return np.load(params_file)['prod_M2_X']

    docs_file_data = np.load(docs_file)
    params_file_data = np.load(params_file)

    docs = docs_file_data['docs']
    vocab_size = docs_file_data['vocab_size']
    n_docs = len(docs)

    alpha0 = params_file_data['alpha0']
    k = params_file_data['k']

    M1 = compute_M1_dist()

    # Krylov method
    k_krylov = np.min([k + 4, vocab_size, 2 * k])
    X = np.random.randn(vocab_size, k_krylov)
    for i in range(2 * q + 1):
        Y = multiply_M2_X_dist(X)
        X = scipy.linalg.qr(Y, mode='economic')

    Y = multiply_M2_X_dist(X)
    U, s, _ = scipy.linalg.svd(Y.T.dot(Y))

    return Q.dot(U), np.sqrt(s)
