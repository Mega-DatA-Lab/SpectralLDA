''' Spectral LDA based on tensor CP Decomposition in distributed mode

Example:

    alpha, beta = spectral_lda(docs, alpha0, k)


REFERENCE
-----------
    https://github.com/Mega-DatA-Lab/SpectralLDA-MXNet/blob/master/report.pdf

'''
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from partitioned_data import pmeta
from rand_svd_dist import rand_svd_dist
from cumulants_dist import moment1_dist, whiten_m3_dist
import mxnet as mx
from proj_l1_simplex import proj_l1_simplex


def spectral_lda_dist(docs, alpha0, k, output_prefix):
    ''' Spectral LDA in distributed mode

    Perform Spectral LDA based on tensor CP Decomposition.

    PARAMETERS
    -----------
    docs : str
        Path for the entire collection of word count vectors.
    alpha0 : float
        Sum of Dirichlet prior parameter.
    k : int
        Rank for the truncated SVD, >= 1.
    output_prefix : str
        Prefix of output files to save alpha and beta. Not a directory.

    RETURNS
    -----------
    alpha : length-k array
        Fitted Dirichlet prior parameter for LDA.
    beta : vocab_size-by-k array
        Fitted topic-word distribution for LDA.
    '''
    # pylint: disable=too-many-locals
    def factor_correct_sign(factors, eps=1e-6):
        ''' Return unique factor matrix with correct signs '''
        _, k = factors[0].shape
        factor = np.zeros_like(factors[0])

        for j in range(k):
            if np.linalg.norm(factors[1][:, j] - factors[2][:, j]) < eps:
                factor[:, j] = factors[0][:, j]
            elif np.linalg.norm(factors[0][:, j] - factors[2][:, j]) < eps:
                factor[:, j] = factors[1][:, j]
            elif np.linalg.norm(factors[0][:, j] - factors[1][:, j]) < eps:
                factor[:, j] = factors[2][:, j]
            else:
                raise RuntimeError('Invalid results from CPDecomp.')

        return factor

    n_docs, vocab_size, _ = pmeta(docs)
    assert n_docs >= 1 and vocab_size >= 1
    assert alpha0 > 0
    assert k >= 1
    assert not Path(output_prefix).is_dir(), ('output_prefix cannot be'
                                              ' a directory')

    # Whiten scaled M3 with SVD of scaled M2
    docs_m1 = moment1_dist(docs)
    eigval_m2, eigvec_m2 = rand_svd_dist(docs, alpha0, k, docs_m1=docs_m1)

    whn = eigvec_m2.dot(np.diag(1.0 / np.sqrt(eigval_m2)))
    whitened_m3 = whiten_m3_dist(docs, whn, alpha0, docs_m1=docs_m1)
    whitened_m3 *= alpha0 * (alpha0 + 1) * (alpha0 + 2) / 2

    # Perform CP Decomposition on whitened M3
    tensor = mx.sym.Variable('t')
    cpdecomp = mx.operator.symbol.CPDecomp3D(tensor, k=k)

    calc = cpdecomp.bind(mx.cpu(),
                         {'t': mx.nd.array(whitened_m3).reshape((k, k, k))})
    cp_results = calc.forward()

    eigval = cp_results[0].asnumpy()
    factors = [mat.asnumpy().T for mat in cp_results[1:]]

    # Unique factor matrix correcting signs of columns
    # in all factor matrices
    unique_factor = factor_correct_sign(factors)

    # Recompose alpha, beta
    alpha = 1 / eigval ** 2
    beta = (eigvec_m2.dot(np.diag(np.sqrt(eigval_m2)))
            .dot(unique_factor).dot(np.diag(eigval)))
    for j in range(k):
        beta[:, j] = proj_l1_simplex(beta[:, j], 1.0)

    # Return in descending order of alpha
    np.savetxt(output_prefix + '_alpha.csv', alpha[::-1])
    np.savetxt(output_prefix + '_beta.csv', beta[:, ::-1])
    return alpha[::-1], beta[:, ::-1]

def main():
    ''' Parse command line and call spectral_lda_dist() '''
    parser = ArgumentParser(description='Spectral LDA')
    parser.add_argument('docs', type=str,
                        help='Path to documents, stored in partitions.')
    parser.add_argument('alpha0', type=float,
                        help='Sum of Dirichlet prior parameter.')
    parser.add_argument('k', type=int,
                        help='Number of topics to extract.')
    parser.add_argument('prefix', type=str,
                        help=('Prefix of output files to save fitted'
                              ' alpha and beta. Cannot be a directory.'))

    args = parser.parse_args()
    spectral_lda_dist(args.docs, args.alpha0, args.k, args.prefix)

if __name__ == '__main__':
    main()
