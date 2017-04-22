''' Test Spectral LDA '''

import numpy as np
from spectral_lda import spectral_lda
from test_cumulants import simulate_word_count_vectors

def test_spectral_lda():
    ''' Simple test cases '''
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
        alpha, beta = spectral_lda(docs, alpha0, k,
                                   n_partitions=n_partitions)

        print('Generative alpha:')
        print(gen_alpha)
        print('Fitted alpha:')
        print(alpha)

        print('Generative beta:')
        print(gen_beta)
        print('Fitted beta:')
        print(beta)

        assert np.all(np.linalg.norm(gen_beta[:, :k] - beta, axis=0) < 0.2)
