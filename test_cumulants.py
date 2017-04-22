''' Test cumulants computations in local mode

'''
from functools import reduce
from operator import add
import numpy as np
import scipy.sparse
import scipy.stats
from cumulants import prod_m2_x, whiten_m3

def simulate_word_count_vectors(alpha, beta, n_docs,
                                min_total_word_count, max_total_word_count):
    ''' Simulate a collection of word count vectors '''
    # Number of topics
    k = len(alpha)
    vocab_size, _k = beta.shape

    assert k == _k
    assert np.allclose(np.sum(beta, axis=0), 1)
    assert n_docs > 0 and vocab_size > 0
    assert max_total_word_count > min_total_word_count
    assert min_total_word_count > 0

    # n_docs-by-k topic assignments
    topic_assignments = scipy.stats.dirichlet.rvs(alpha, size=n_docs)

    # length-n_docs vector of total word counts per document
    total_word_counts = np.random.randint(min_total_word_count,
                                          max_total_word_count,
                                          size=n_docs)

    word_count_vectors = np.zeros((n_docs, vocab_size))
    for i, (total_word_count, topic_assignment) \
        in enumerate(zip(total_word_counts, topic_assignments)):
        word_count_vectors[i, :] = scipy.stats.multinomial.rvs(
            total_word_count, beta.dot(topic_assignment))

    return word_count_vectors

def compute_e2(word_count_vectors, n_docs):
    ''' Compute E2 by summing the contribution of every document '''
    def doc_contrib(word_count_vector):
        ''' Compute contribution of the current document to E2 '''
        total_count = np.sum(word_count_vector)
        assert total_count >= 2
        return ((np.outer(word_count_vector, word_count_vector)
                 - np.diag(word_count_vector))
                / total_count / (total_count - 1) / n_docs)

    return reduce(add, [doc_contrib(word_count_vector)
                        for word_count_vector in word_count_vectors])

def compute_e3(word_count_vectors, n_docs):
    ''' Compute E3 by summing the contribution of every document '''
    def doc_contrib(word_count_vector):
        ''' Compute contribution of the current document to E3 '''
        total_count = np.sum(word_count_vector)
        assert total_count >= 3
        contrib = np.einsum('i,j,k->ijk', word_count_vector,
                            word_count_vector, word_count_vector)

        for i, wc_i in enumerate(word_count_vector):
            for j, wc_j in enumerate(word_count_vector):
                for k, wc_k in enumerate(word_count_vector):
                    if i == j:
                        contrib[i, j, k] -= wc_i * wc_k
                    if j == k:
                        contrib[i, j, k] -= wc_i * wc_j
                    if k == i:
                        contrib[i, j, k] -= wc_j * wc_k
                    if i == j and j == k:
                        contrib[i, j, k] += 2 * wc_i

        return (contrib / total_count / (total_count - 1)
                / (total_count - 2) / n_docs)

    return reduce(add, [doc_contrib(word_count_vector)
                        for word_count_vector in word_count_vectors])

def compute_m1(word_count_vectors):
    ''' Compute M1 i.e. average of the normalised word count vectors '''
    normalized = word_count_vectors.T / np.sum(word_count_vectors, axis=1)
    return np.mean(normalized.T, axis=0)

def compute_m2(word_count_vectors, alpha0, n_docs):
    ''' Compute M2 '''
    docs_m1 = compute_m1(word_count_vectors)
    docs_e2 = compute_e2(word_count_vectors, n_docs)

    return docs_e2 - alpha0 / (alpha0 + 1) * np.outer(docs_m1, docs_m1)

def compute_m3(word_count_vectors, alpha0, n_docs):
    ''' Compute M3 '''
    docs_m1 = compute_m1(word_count_vectors)
    docs_e2 = compute_e2(word_count_vectors, n_docs)
    docs_e3 = compute_e3(word_count_vectors, n_docs)

    assert np.isclose(np.sum(docs_e3), 1)

    adj_e2 = (np.einsum('ij,k->ijk', docs_e2, docs_m1)
              + np.einsum('ij,k->ikj', docs_e2, docs_m1)
              + np.einsum('ij,k->kij', docs_e2, docs_m1))
    adj_m1 = np.einsum('i,j,k->ijk', docs_m1, docs_m1, docs_m1)

    return (docs_e3 - alpha0 / (alpha0 + 2) * adj_e2
            + 2 * alpha0 ** 2 / (alpha0 + 1) / (alpha0 + 2) * adj_m1)

def contract_m3(ts3, test_x):
    ''' Contract each dimension of ts3 by multiplying by test_x.T '''
    contracted_d0 = np.einsum('ij,jkl->ikl', test_x.T, ts3)
    contracted_d1 = np.einsum('ij,kjl->kil', test_x.T, contracted_d0)
    contracted_d2 = np.einsum('ij,klj->kli', test_x.T, contracted_d1)

    return contracted_d2

def test_random_product_m2_x():
    ''' Test product of M2 by random test matrix X '''
    k = 10
    vocab_size = 100
    n_docs = 200

    alpha = [5] * k
    beta = np.random.rand(vocab_size, k)
    beta /= beta.sum(axis=0)

    min_total_word_count = 500
    max_total_word_count = 1001

    word_count_vectors = simulate_word_count_vectors(alpha, beta, n_docs,
                                                     min_total_word_count,
                                                     max_total_word_count)

    alpha0 = np.sum(alpha)
    docs_m2 = compute_m2(word_count_vectors, alpha0, n_docs)

    test_x = np.random.randn(vocab_size, k)
    m2x = prod_m2_x(word_count_vectors, test_x, alpha0,
                    n_partitions=3)

    assert np.linalg.norm(docs_m2.dot(test_x) - m2x) <= 1e-6

def test_whiten_m3():
    ''' Test whitening M3 by random test matrix X '''
    k = 5
    vocab_size = 20
    n_docs = 50

    alpha = [5] * k
    beta = np.random.rand(vocab_size, k)
    beta /= beta.sum(axis=0)

    min_total_word_count = 500
    max_total_word_count = 1001

    word_count_vectors = simulate_word_count_vectors(alpha, beta, n_docs,
                                                     min_total_word_count,
                                                     max_total_word_count)

    alpha0 = np.sum(alpha)
    docs_m3 = compute_m3(word_count_vectors, alpha0, n_docs)

    test_x = np.random.randn(vocab_size, k)
    whitened_m3 = (whiten_m3(word_count_vectors, test_x, alpha0,
                             n_partitions=3)
                   .reshape((k, k, k)))

    assert np.linalg.norm(contract_m3(docs_m3, test_x) - whitened_m3) <= 1e-6
