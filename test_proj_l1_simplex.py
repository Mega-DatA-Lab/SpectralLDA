''' Test projection onto l1-Simplex '''

import numpy as np
from proj_l1_simplex import proj_l1_simplex

def test_proj_l1_simplex():
    ''' Simple test cases '''
    vec1 = np.array([2, 5, 3, 7])
    solutions = [[0., 0., 0., 2.],
                 [0., 1., 0., 3.],
                 [0., 2., 0., 4.],
                 [0., 2.66666667, 0.66666667, 4.66666667],
                 [0.25, 3.25, 1.25, 5.25],
                 [0.75, 3.75, 1.75, 5.75],
                 [1.25, 4.25, 2.25, 6.25],
                 [1.75, 4.75, 2.75, 6.75],
                 [2.25, 5.25, 3.25, 7.25]]

    for l1_simplex_boundary, solution in zip(range(2, 20, 2), solutions):
        assert np.allclose(solution,
                           proj_l1_simplex(vec1, l1_simplex_boundary))
