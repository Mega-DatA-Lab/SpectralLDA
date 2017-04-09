''' Test cumulants computations in local mode

'''
import numpy as np
import scipy.sparse
import scipy.stats
from functools import reduce
from operator import add
from cumulants import *

def simulate_word_count_vectors(alpha, beta, n_docs,
                                min_total_word_count,
                                max_total_word_count,
                                vocab_size):
    ''' Simulate a collection of word count vectors

    '''
    # Number of topics
    k = len(alpha)

    assert (vocab_size, k) == beta.shape
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
    ''' Compute E2 by summing the contribution of every document

    '''
    def doc_contrib(wc):
        l = np.sum(wc)
        assert l >= 2
        return (np.outer(wc, wc) - np.diag(wc)) / l / (l - 1) / n_docs

    return reduce(add, [doc_contrib(wc) for wc in word_count_vectors])

def compute_e3(word_count_vectors, n_docs):
    ''' Compute E3 by summing the contribution of every document

    '''
    def doc_contrib(wc):
        l = np.sum(wc)
        assert l >= 3
        contrib = np.einsum('i,j,k->ijk', wc, wc, wc)

        for i, wc_i in enumerate(wc):
            for j, wc_j in enumerate(wc):
                for k, wc_k in enumerate(wc):
                    if i == j:
                        contrib[i, j, k] -= wc_i * wc_k
                    if j == k:
                        contrib[i, j, k] -= wc_i * wc_j
                    if k == i:
                        contrib[i, j, k] -= wc_j * wc_k
                    if i == j and j == k:
                        contrib[i, j, k] += 2 * wc_i

        return contrib / l / (l - 1) / (l - 2) / n_docs

    return reduce(add, [doc_contrib(wc) for wc in word_count_vectors])

def compute_m1(word_count_vectors):
    ''' Compute M1 i.e. the average of the normalised word count vectors

    '''
    normalized = word_count_vectors.T / np.sum(word_count_vectors, axis=1)
    return np.mean(normalized.T, axis=0)

def compute_m2(word_count_vectors, alpha0, n_docs):
    ''' Compute M2 by subtracting from E2 adjustment term related to M1

    '''
    m1 = compute_m1(word_count_vectors)
    e2 = compute_e2(word_count_vectors, n_docs)

    return e2 - alpha0 / (alpha0 + 1) * np.outer(m1, m1)

def compute_m3(word_count_vectors, alpha0, n_docs):
    ''' Compute M3 by subtracting from E3 adjustment terms related to M1 and E2

    '''
    m1 = compute_m1(word_count_vectors)
    e2 = compute_e2(word_count_vectors, n_docs)
    e3 = compute_e3(word_count_vectors, n_docs)

    assert np.isclose(np.sum(e3), 1)

    adj_e2 = (np.einsum('ij,k->ijk', e2, m1)
              + np.einsum('ij,k->ikj', e2, m1)
              + np.einsum('ij,k->kij', e2, m1))
    adj_m1 = np.einsum('i,j,k->ijk', m1, m1, m1)

    return (e3 - alpha0 / (alpha0 + 2) * adj_e2
            + 2 * alpha0 ** 2 / (alpha0 + 1) / (alpha0 + 2) * adj_m1)

def contract_m3(m3, x):
    ''' Contract each dimension of m3 by multiplying by x.T

    '''
    contracted_d0 = np.einsum('ij,jkl->ikl', x.T, m3)
    contracted_d1 = np.einsum('ij,kjl->kil', x.T, contracted_d0)
    contracted_d2 = np.einsum('ij,klj->kli', x.T, contracted_d1)

    return contracted_d2

def test_random_product_m2_x():
    ''' Test product of M2 by random test matrix X

    '''
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
                                                     max_total_word_count,
                                                     vocab_size)

    alpha0 = np.sum(alpha)
    m1 = compute_m1(word_count_vectors)
    m2 = compute_m2(word_count_vectors, alpha0, n_docs)

    x = np.random.randn(vocab_size, k)
    prod_m2_x = mult_M2_X(word_count_vectors, x, n_docs, m1, alpha0)

    assert np.linalg.norm(m2.dot(x) - prod_m2_x) <= 1e-6

def test_whiten_m3():
    ''' Test whitening M3 by random test matrix X

    '''
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
                                                     max_total_word_count,
                                                     vocab_size)

    alpha0 = np.sum(alpha)
    m1 = compute_m1(word_count_vectors)
    m3 = compute_m3(word_count_vectors, alpha0, n_docs)

    x = np.random.randn(vocab_size, k)
    whitened_m3 = (whiten_M3(word_count_vectors, x, n_docs, m1, alpha0)
                   .reshape((k, k, k)))

    assert np.linalg.norm(contract_m3(m3, x) - whitened_m3) <= 1e-6
