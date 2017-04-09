import numpy as np
import scipy.sparse as spsp


def mult_E2_X(docs, X, n_docs):
    '''Compute the product of E[x1\odot x2] and X

    The contribution of each document in ``docs`` is

    .. math::

        \frac{1}{n_docs}\left\{\sum_{i=1}^m\frac{1}{l_i(l_i-1)}\left(p_i\odot p_i-\text{diag}(p_i)\right)\right\},

    where :math:`l_i` and :math:`p_i` are the length and the word count vector of the :math:`i`-th document in ``docs``.

    Parameters
    -----------
    docs : m-by-V csr_matrix
        m word count vectors, where V is the vocabulary size, m could be smaller than the total number of documents
    X : V-by-k array
        where k is the number of factors
    n_docs : integer
        The total number of documents

    Returns
    ----------
    out : V-by-k csr_matrix
        Sum of the contributions of each document to :math:`E[x_1\odot x_2]`
    '''
    l = np.squeeze(np.array(docs.sum(axis=1)))
    diag_l = spsp.diags(1.0 / l / (l - 1))

    scaled_D = diag_l.dot(docs)
    t1 = scaled_D.T.dot(docs.dot(X))

    sum_scaled_D = np.squeeze(np.array(scaled_D.sum(axis=0)))
    t2 = spsp.diags(sum_scaled_D).dot(X)

    return (t1 - t2) / n_docs

def mult_M2_X_helper(prod_E2_X, X, M1, alpha0):
    ''' Compute the product of M2 by X


    '''
    prod_M1_X = alpha0 / (alpha0 + 1) * np.outer(M1, M1.dot(X))
    return prod_E2_X - prod_M1_X

def mult_M2_X(docs, X, n_docs, M1, alpha0):
    prod_E2_X = mult_E2_X(docs, X, n_docs)
    return mult_M2_X_helper(prod_E2_X, X, M1, alpha0)

def whiten_E3(docs, W, n_docs):
    m, _ = docs.shape
    _, k = W.shape

    # m-by-k
    p = docs.dot(W)

    # length-m
    l = np.squeeze(np.array(docs.sum(axis=1)))
    diag_l = spsp.diags(1.0 / l / (l - 1) / (l - 2))

    # m-by-k
    scaled_docs = diag_l.dot(docs)
    scaled_p = diag_l.dot(p)
    # V-by-k
    r = scaled_docs.T.dot(p)
    vocab_size, _ = r.shape
    # length-V
    r2 = np.squeeze(np.array(scaled_docs.sum(axis=0)))

    # ======== whiten E3 ========
    # k-by-k-by-k
    outer_p_p = np.einsum('ij,ik->ijk', p, p).reshape((m, -1))
    t1 = scaled_p.T.dot(outer_p_p).flatten()

    outer_w_w = np.einsum('ij,ik->ijk', W, W).reshape((vocab_size, -1))
    outer_r_w = np.einsum('ij,ik->ijk', r, W).reshape((vocab_size, -1))
    t21 = np.einsum('ij,ik->ijk', r, outer_w_w).reshape((vocab_size, -1))
    t22 = np.einsum('ij,ik->ijk', W, outer_r_w).reshape((vocab_size, -1))
    t23 = np.einsum('ij,ik->ijk', outer_w_w, r).reshape((vocab_size, -1))

    t2 = t21.sum(axis=0) + t22.sum(axis=0) + t23.sum(axis=0)

    outer_w_3 = np.einsum('ij,ik,il->ijkl', W, W, W).reshape((vocab_size, -1))
    t3 = 2 * r2.dot(outer_w_3)

    whitened_E3 = (t1 - t2 + t3).reshape((k, k * k)) / n_docs
    return whitened_E3

def whiten_E2_M1(docs, W, n_docs, M1):
    m, _ = docs.shape
    _, k = W.shape

    # m-by-k
    p = docs.dot(W)
    # length-k
    q = M1.dot(W)

    # length-m
    l = np.squeeze(np.array(docs.sum(axis=1)))
    diag_l = spsp.diags(1.0 / l / (l - 1))

    # m-by-V
    scaled_docs = diag_l.dot(docs)
    # m-by-k
    scaled_p = diag_l.dot(p)
    # V-by-k
    r = np.outer(scaled_docs.sum(axis=0), q)
    vocab_size, _ = r.shape

    tiled_q = np.tile(q, [m, 1])

    # ====== whiten E2_M1 =========
    outer_scaled_p_p = np.einsum('ij,ik->ijk', scaled_p, p).reshape((m, -1))
    outer_q_scaled_p = (np.einsum('ij,ik->ijk', tiled_q, scaled_p)
                        .reshape((m, -1)))
    outer_w_w = np.einsum('ij,ik->ijk', W, W).reshape((vocab_size, -1))
    outer_r_w = np.einsum('ij,ik->ijk', r, W).reshape((vocab_size, -1))

    u11 = np.einsum('ij,ik->ijk', tiled_q, outer_scaled_p_p).reshape((m, -1))
    u12 = np.einsum('ij,ik->ijk', p, outer_q_scaled_p).reshape((m, -1))
    u13 = np.einsum('ij,ik->ijk', outer_scaled_p_p, tiled_q).reshape((m, -1))

    u21 = np.einsum('ij,ik->ijk', r, outer_w_w).reshape((vocab_size, -1))
    u22 = np.einsum('ij,ik->ijk', W, outer_r_w).reshape((vocab_size, -1))
    u23 = np.einsum('ij,ik->ijk', outer_w_w, r).reshape((vocab_size, -1))

    u1 = u11.sum(axis=0) + u12.sum(axis=0) + u13.sum(axis=0)
    u2 = u21.sum(axis=0) + u22.sum(axis=0) + u23.sum(axis=0)

    whitened_E2_M1 = ((u1 - u2) / n_docs).reshape((k, k * k))
    return whitened_E2_M1

def whiten_M3_helper(whitened_E3, whitened_E2_M1, W, M1, alpha0):
    ''' Whiten M3

    '''
    vocab_size = len(M1)
    _, k = W.shape
    q = M1.dot(W)
    whitened_M1_3 = np.einsum('i,j,k->ijk', q, q, q).reshape((k, k * k))

    whitened_M3 = (
        whitened_E3
        - alpha0 / (alpha0 + 2) * whitened_E2_M1
        + 2 * alpha0 ** 2 / (alpha0 + 1) / (alpha0 + 2) * whitened_M1_3
    )

    return whitened_M3

def whiten_M3(docs, W, n_docs, M1, alpha0):
    whitened_E3 = whiten_E3(docs, W, n_docs)
    whitened_E2_M1 = whiten_E2_M1(docs, W, n_docs, M1)
    return whiten_M3_helper(whitened_E3, whitened_E2_M1,
                            W, M1, alpha0)

