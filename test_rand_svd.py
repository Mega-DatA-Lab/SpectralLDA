''' Test Randomised SVD for M2 '''

import numpy as np
import scipy.linalg
from rand_svd import rand_svd
from test_cumulants import simulate_word_count_vectors

def test_rand_svd():
    ''' Simple test cases '''
    # pylint: disable=too-many-locals
    gen_alpha = [10, 5, 2]
    gen_k = len(gen_alpha)

    vocab_size = 10
    n_docs = 2000

    gen_beta = np.random.rand(vocab_size, gen_k)
    gen_beta /= gen_beta.sum(axis=0)

    docs = simulate_word_count_vectors(gen_alpha, gen_beta, n_docs, 200, 400)

    for n_partitions in [1, 3]:
        k = gen_k
        alpha0 = np.sum(gen_alpha[:k])
        eigval_m2, eigvec_m2 = rand_svd(docs, alpha0, k, n_iter=3,
                                        n_partitions=n_partitions)

        scaled_m2 = (gen_beta[:, :k].dot(np.diag(gen_alpha[:k]))
                     .dot(gen_beta[:, :k].T))
        svd_u, svd_s, _ = scipy.linalg.svd(scaled_m2)

        assert np.linalg.norm(svd_s[:k] - eigval_m2) < 0.1
        for j in range(k):
            assert (min(np.linalg.norm(svd_u[:, j] - eigvec_m2[:, j]),
                        np.linalg.norm(svd_u[:, j] + eigvec_m2[:, j])) < 0.6)
