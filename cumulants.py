''' Cumulants computations in local mode

It computes M1, the product of M2 with a test matrix, and whitening
of M3, over partitioned documents.

`contrib_xxx()` computes contribution of the current partition
to the moment, the product of the moment with a test matrix, or
whitened moment.

`xxx_adj()` computes the list of adjustment terms for the product
of the moment with a test matrix, or whitened moment.
'''
from functools import reduce
from operator import add
import numpy as np
import scipy.sparse as spsp


def equal_partitions(n_docs, n_partitions):
    ''' Compute partition ranges over [0, n_docs)

    Returns an iterator over (start, end) tuples such that
    range(start, end) marks the index range of each partition.

    Parameters
    -----------
    n_docs : int
        Total number of documents.

    Returns
    -----------
    out : iterator over (start, end) tuples
        Iterator over tuples such that range(start, end) marks
        the index range of each partition.
    '''
    assert n_docs >= 1 and n_partitions >= 1

    starts = [i * (n_docs // n_partitions) for i in range(n_partitions)]
    ends = starts[1:] + [n_docs]
    return zip(starts, ends)

# ================= M1 Calculation =================
def contrib_m1(docs, n_docs):
    ''' Compute contribution of the partition to M1

    Parameters
    -----------
    docs : m-by-vocab_size array or csr_matrix
        Current partition of word count vectors.
    n_docs : int
        total number of documents

    Returns
    ----------
    out : length-V array
        contribution of `docs` to M1
    '''
    partition_size, vocab_size = docs.shape
    assert partition_size >= 1 and vocab_size >= 1
    assert n_docs >= partition_size

    # transposed normalised docs
    _docs = docs.T / np.squeeze(docs.sum(axis=1))
    _docs = _docs.T

    assert np.allclose(_docs.sum(axis=1), 1)
    return np.squeeze(np.array(_docs.mean(axis=0))) * (docs.shape[0] / n_docs)

def moment1(docs, n_partitions=1):
    ''' Compute M1 over partitioned documents

    Compute over partitioned documents in local mode, mostly used
    for test purpose.

    Parameters
    -----------
    docs : n_docs-by-vocab_size array or csr_matrix
        The entire collection of word count vectors.
    n_partitions : int
        Number of partitions over the document collection, >= 1.

    Returns
    ----------
    out : length-vocab_size array
        M1 of the entire document collection
    '''
    n_docs, vocab_size = docs.shape
    assert n_docs >= 1 and vocab_size >= 1
    assert n_partitions >= 1 and n_partitions <= n_docs

    contribs = [contrib_m1(docs[start:end, :], n_docs)
                for start, end in equal_partitions(n_docs, n_partitions)]

    return reduce(add, contribs)


# ================ M2 Calculation ================
def contrib_prod_e2_x(docs, test_x, n_docs):
    ''' Compute contribution of the partition to the product of E2 and X

    Parameters
    -----------
    docs : m-by-vocab_size array or csr_matrix
        Current partition of word count vectors.
    test_x : vocab_size-by-k array
        Test matrix where k is the number of factors.
    n_docs : int
        Total number of documents, could be greater than m.

    Returns
    ----------
    out : vocab_size-by-k array
        Contribution of the partition to the product of E2 and X.
    '''
    partition_size, vocab_size = docs.shape
    _vocab_size, num_factors = test_x.shape
    assert partition_size >= 1 and vocab_size >= 1
    assert partition_size <= n_docs
    assert vocab_size == _vocab_size and num_factors >= 1

    document_lengths = np.squeeze(np.array(docs.sum(axis=1)))
    diag_l = spsp.diags(1.0 / document_lengths / (document_lengths - 1))

    scaled_docs = diag_l.dot(docs)
    prod_x = scaled_docs.T.dot(docs.dot(test_x))

    sum_scaled_docs = np.squeeze(np.array(scaled_docs.sum(axis=0)))
    prod_x_adj = spsp.diags(sum_scaled_docs).dot(test_x)

    return (prod_x - prod_x_adj) / n_docs

def prod_m2_x(docs, test_x, alpha0, docs_m1=None, n_partitions=1):
    ''' Compute the product of M2 by test matrix X over partitions of documents

    Compute over partitions of documents in local mode, mostly
    used for test purpose.

    Parameters
    -----------
    docs : n_docs-by-vocab_size array
        Entire collection of word count vectors.
    test_x : vocab_size-by-k array
        Test matrix where k is the number of factors.
    alpha0 : float
        Sum of the Dirichlet prior parameter.
    docs_m1: length-vocab_size array, optional
        M1 of the entire collection of word count vectors.
    n_partitions : int, optional
        Number of partitions, 1 by default.

    Returns
    -----------
    out : vocab_size-by-k array
        Product of M2 by X.
    '''
    def reduce_and_adjust(contribs, docs_m1, test_x, alpha0):
        ''' Reduce contributions and adjust '''
        adj = alpha0 / (alpha0 + 1) * np.outer(docs_m1, docs_m1.dot(test_x))
        return reduce(add, contribs) - adj

    n_docs, vocab_size = docs.shape
    _vocab_size, num_factors = test_x.shape
    assert n_docs >= 1 and vocab_size >= 1
    assert n_partitions >= 1 and n_partitions <= n_docs
    assert vocab_size == _vocab_size and num_factors >= 1
    if docs_m1 is not None:
        assert docs_m1.ndim == 1 and vocab_size == len(docs_m1)
    assert alpha0 > 0

    contribs = [contrib_prod_e2_x(docs[start:end, :], test_x, n_docs)
                for start, end in equal_partitions(n_docs, n_partitions)]

    if docs_m1 is None:
        docs_m1 = moment1(docs, n_partitions)
    return reduce_and_adjust(contribs, docs_m1, test_x, alpha0)


# ============== M3 calculation ================
def contrib_whiten_e3(docs, whn, n_docs):
    ''' Compute contribution of the partition to whitening E3

    Parameters
    -----------
    docs : m-by-vocab_size array or csr_matrix
        Current partition of word count vectors.
    whn : vocab_size-by-k array
        Whitening matrix.
    n_docs : int
        Total number of documents, could be greater than m.

    Returns
    ----------
    out : k-by-(k ** 2) array
        Contribution to the whitened E3, unfolded version.
    '''
    # pylint: disable=too-many-locals
    partition_size, vocab_size = docs.shape
    _vocab_size, num_factors = whn.shape
    assert partition_size >= 1 and vocab_size >= 1
    assert vocab_size == _vocab_size and num_factors >= 1
    assert partition_size <= n_docs

    # m-by-k
    whitened_docs = docs.dot(whn)

    # length-m
    document_lengths = np.squeeze(np.array(docs.sum(axis=1)))
    diag_l = spsp.diags(1.0 / document_lengths
                        / (document_lengths - 1) / (document_lengths - 2))

    # m-by-k
    scaled_docs = diag_l.dot(docs)
    scaled_whitened_docs = diag_l.dot(whitened_docs)

    # V-by-k
    # The 3rd vector for the tensor products w w rho and its
    # permutations, where w is every row of the whitening matrix.
    # We reduce over the current partition to obtain rho.
    rho = scaled_docs.T.dot(whitened_docs)

    # length-V
    # Coefficient vector before the tensor products w w w,
    # where w is every row of the whitening matrix.
    tau = np.squeeze(np.array(scaled_docs.sum(axis=0)))

    # ======== whiten E3 ========
    # 1st-order terms
    # k-by-k-by-k
    outer_p_p = (np.einsum('ij,ik->ijk', whitened_docs, whitened_docs)
                 .reshape((partition_size, -1)))
    terms1 = scaled_whitened_docs.T.dot(outer_p_p).flatten()

    # 2nd-order terms
    outer_w_w = np.einsum('ij,ik->ijk', whn, whn).reshape((vocab_size, -1))
    outer_r_w = np.einsum('ij,ik->ijk', rho, whn).reshape((vocab_size, -1))
    t21 = np.einsum('ij,ik->ijk', rho, outer_w_w).reshape((vocab_size, -1))
    t22 = np.einsum('ij,ik->ijk', whn, outer_r_w).reshape((vocab_size, -1))
    t23 = np.einsum('ij,ik->ijk', outer_w_w, rho).reshape((vocab_size, -1))

    terms2 = t21.sum(axis=0) + t22.sum(axis=0) + t23.sum(axis=0)

    # 3rd-order terms
    outer_w_3 = (np.einsum('ij,ik,il->ijkl', whn, whn, whn)
                 .reshape((vocab_size, -1)))
    terms3 = 2 * tau.dot(outer_w_3)

    whitened_e3 = (terms1 - terms2 + terms3) / n_docs
    return whitened_e3.reshape((num_factors, -1))

def contrib_whiten_e2m1(docs, docs_m1, whn, n_docs):
    ''' Compute contribution of the partition to whitening the tensor products of E2 and M1

    Parameters
    -----------
    docs : m-by-vocab_size array or csr_matrix
        Current partition of word count vectors.
    docs_m1 : length-vocab_size array
        M1 (average word count vector) of the entire collection
        of word count vectors.
    whn : vocab_size-by-k array
        Whitening matrix.
    n_docs : int
        Total number of documents, could be greater than m.

    Returns
    ----------
    out : k-by-(k ** 2) array
        Contribution to the whitened tensor products of E2 and M1,
        unfolded version.
    '''
    # pylint: disable=too-many-locals
    partition_size, vocab_size = docs.shape
    _vocab_size, num_factors = whn.shape
    assert partition_size >= 1 and vocab_size >= 1
    assert n_docs >= partition_size
    assert vocab_size == _vocab_size and num_factors >= 1

    # m-by-k
    whitened_docs = docs.dot(whn)
    # length-k
    whitened_m1 = docs_m1.dot(whn)
    tiled_whitened_m1 = np.tile(whitened_m1, [partition_size, 1])

    # length-m
    document_lengths = np.squeeze(np.array(docs.sum(axis=1)))
    diag_l = spsp.diags(1.0 / document_lengths / (document_lengths - 1))

    # m-by-V
    scaled_docs = diag_l.dot(docs)
    # m-by-k
    scaled_whitened_docs = diag_l.dot(whitened_docs)

    # V-by-k
    # The 3rd vector to make tensor products w w rho and its
    # permutations, where w is every row of the whitening matrix.
    # We reduce over the partition to obtain r1.
    rho = np.outer(scaled_docs.sum(axis=0), whitened_m1)

    # ====== whiten E2_M1 =========
    # 1st-order terms
    outer_scaled_p_p = (np.einsum('ij,ik->ijk', scaled_whitened_docs,
                                  whitened_docs)
                        .reshape((partition_size, -1)))
    outer_q_scaled_p = (np.einsum('ij,ik->ijk', tiled_whitened_m1,
                                  scaled_whitened_docs)
                        .reshape((partition_size, -1)))

    u11 = (np.einsum('ij,ik->ijk', tiled_whitened_m1, outer_scaled_p_p)
           .reshape((partition_size, -1)))
    u12 = (np.einsum('ij,ik->ijk', whitened_docs, outer_q_scaled_p)
           .reshape((partition_size, -1)))
    u13 = (np.einsum('ij,ik->ijk', outer_scaled_p_p, tiled_whitened_m1)
           .reshape((partition_size, -1)))

    whitened = u11.sum(axis=0) + u12.sum(axis=0) + u13.sum(axis=0)

    # 2nd-order terms
    outer_w_w = np.einsum('ij,ik->ijk', whn, whn).reshape((vocab_size, -1))
    outer_r_w = np.einsum('ij,ik->ijk', rho, whn).reshape((vocab_size, -1))

    u21 = np.einsum('ij,ik->ijk', rho, outer_w_w).reshape((vocab_size, -1))
    u22 = np.einsum('ij,ik->ijk', whn, outer_r_w).reshape((vocab_size, -1))
    u23 = np.einsum('ij,ik->ijk', outer_w_w, rho).reshape((vocab_size, -1))

    whitened_adj = u21.sum(axis=0) + u22.sum(axis=0) + u23.sum(axis=0)

    whitened_e2_m1 = (whitened - whitened_adj) / n_docs
    return whitened_e2_m1.reshape((num_factors, -1))

def whiten_m3(docs, whn, alpha0, docs_m1=None, n_partitions=1):
    ''' Whiten M3

    Compute over partition of documents in local mode, mostly used
    for test purpose.

    Parameters
    -----------
    docs : n_docs-by-vocab_size array or csr_matrix
        Entire collection of word count vectors.
    whn : vocab_size-by-k array
        Whitening matrix.
    alpha0 : float
        Sum of Dirichlet prior parameter.
    docs_m1 : length-vocab_size array, optional
        M1 of the entire collection of word count vectors.
    n_partitions : int, optional
        Number of partitions, 1 by default.

    Returns
    ----------
    out : k-by-(k ** 2) array
        Whitened M3, unfolded version.
    '''
    def reduce_and_adjust(contribs_e3, contribs_e2m1, docs_m1, whn, alpha0):
        ''' Reduce the contributions and adjust '''
        _, num_factors = whn.shape
        # length-k
        whitened_m1 = docs_m1.dot(whn)
        whitened_m1_3 = (np.einsum('i,j,k->ijk', whitened_m1, whitened_m1,
                                   whitened_m1).reshape((num_factors, -1)))

        coeff1 = alpha0 / (alpha0 + 2)
        coeff2 = 2 * alpha0 ** 2 / (alpha0 + 1) / (alpha0 + 2)
        return (reduce(add, contribs_e3)
                - coeff1 * reduce(add, contribs_e2m1)
                + coeff2 * whitened_m1_3)

    n_docs, vocab_size = docs.shape
    _vocab_size, num_factors = whn.shape
    assert n_docs >= 1 and vocab_size >= 1
    assert n_partitions <= n_docs
    assert vocab_size == _vocab_size and num_factors >= 1
    if docs_m1 is not None:
        assert docs_m1.ndim == 1 and vocab_size == len(docs_m1)
    assert alpha0 > 0

    contribs_e3 = [contrib_whiten_e3(docs[start:end, :], whn, n_docs)
                   for start, end in equal_partitions(n_docs, n_partitions)]

    if docs_m1 is None:
        docs_m1 = moment1(docs, n_partitions)
    contribs_e2m1 = [contrib_whiten_e2m1(docs[start:end, :], docs_m1,
                                         whn, n_docs)
                     for start, end in equal_partitions(n_docs, n_partitions)]

    return reduce_and_adjust(contribs_e3, contribs_e2m1, docs_m1,
                             whn, alpha0)
