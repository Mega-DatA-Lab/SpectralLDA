''' Projection onto the l1-Simplex

Project a given vector onto the l1-Simplex with minimal shift measured by the l2-norm.

Example:

    proj_vec, theta = proj_l1_simplex(vec, l1_simplex_boundary)


REFERENCE
----------

    Duchi, John, Efficient Projections onto the l1-Ball for Learning in High Dimensions.

'''

import numpy as np


def proj_l1_simplex(vec, l1_simplex_boundary):
    ''' Project a vector onto the l1-Simplex

    PARAMETERS
    -----------
    vec : 1d array
        Input vector.
    l1_simplex_boundary : float
        Upper value of the l1-norm.

    RETURNS
    -----------
    out : 1d array of the same shape as vec
        Projected vector.
    theta : float
        Shift as computed by the Duchi algorithm.
    '''
    assert vec.ndim == 1 and len(vec) >= 1
    assert l1_simplex_boundary > 0
    vec_sorted = np.sort(vec)[::-1]

    vec_shifted = (vec_sorted - (vec_sorted.cumsum() - l1_simplex_boundary)
                   / range(1, len(vec) + 1))
    rho = np.squeeze(np.where(vec_shifted > 0)).max() + 1

    theta = (vec_sorted[:rho].sum() - l1_simplex_boundary) / rho

    return np.maximum(vec - theta, 0), theta
