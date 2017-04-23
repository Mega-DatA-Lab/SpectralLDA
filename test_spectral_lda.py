''' Test Spectral LDA '''

import numpy as np
from scipy.sparse import csr_matrix
from spectral_lda import spectral_lda
from test_cumulants import simulate_word_count_vectors

def test_spectral_lda():
    ''' Simple test cases '''
    gen_alpha = [10, 5, 2]
    gen_k = len(gen_alpha)

    vocab_size = 50
    n_docs = 5000

    gen_beta = np.random.rand(vocab_size, gen_k)
    for j in range(gen_k):
        gen_beta[(j * 5):((j + 1) * 5), j] += 4
    gen_beta /= gen_beta.sum(axis=0)

    docs = simulate_word_count_vectors(gen_alpha, gen_beta, n_docs, 500, 1000)

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

def test_spectral_lda_csr_matrix():
    ''' Simple test cases '''
    gen_alpha = [10, 5, 2]
    gen_k = len(gen_alpha)

    vocab_size = 50
    n_docs = 5000

    gen_beta = np.random.rand(vocab_size, gen_k)
    for j in range(gen_k):
        gen_beta[(j * 5):((j + 1) * 5), j] += 4
    gen_beta /= gen_beta.sum(axis=0)

    docs = simulate_word_count_vectors(gen_alpha, gen_beta, n_docs, 500, 1000)
    docs = csr_matrix(docs)

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
