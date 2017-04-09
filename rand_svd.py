''' Compute Randomised SVD in distributed mode

'''
import os
import subprocess
import numpy as np
import scipy.linalg
from utils import sync_dir

global dmlc_workers

def rand_svd_dist(docs_file, params_file, q):

    def launch_compute_M1_dist():
        cmd = '{} --launcher ssh'.format(os.environ['PS_LAUNCH'])
        cmd += ' -n {}'.format(os.environ['DMLC_NUM_WORKER'])
        cmd += ' -s {}'.format(os.environ['DMLC_NUM_SERVER'])
        cmd += ' -H {}'.format(os.environ['DMLC_HOSTFILE'])
        cmd += ' --sync-dst-dir {}'.format(os.environ['LDA_DATA_PATH'])
        cmd += ' python3 cumulants_dist.py compute_M1'
        cmd += ' --data-path {}'.format(os.environ['LDA_DATA_PATH'])

        subprocess.run([cmd])
        sync_dir(os.environ['LDA_DATA_PATH'], dmlc_workers[0],
                 os.environ['LDA_DATA_PATH'])

        return np.load(params_file)['M1']

    def launch_mult_M2_X_dist(X):
        np.savez(params_file, X=X, M1=M1, alpha0=alpha0, k=k)

        cmd = '{} --launcher ssh'.format(os.environ['PS_LAUNCH'])
        cmd += ' -n {}'.format(os.environ['DMLC_NUM_WORKER'])
        cmd += ' -s {}'.format(os.environ['DMLC_NUM_SERVER'])
        cmd += ' -H {}'.format(os.environ['DMLC_HOSTFILE'])
        cmd += ' --sync-dst-dir {}'.format(os.environ['LDA_DATA_PATH'])
        cmd += ' python3 cumulants_dist.py mult_M2_X'
        cmd += ' --data-path {}'.format(os.environ['LDA_DATA_PATH'])

        subprocess.run([cmd])
        sync_dir(os.environ['LDA_DATA_PATH'], dmlc_workers[0],
                 os.environ['LDA_DATA_PATH'])

        return np.load(params_file)['prod_M2_X']

    docs_file_data = np.load(docs_file)
    params_file_data = np.load(params_file)

    docs = docs_file_data['docs']
    vocab_size = docs_file_data['vocab_size']
    n_docs = len(docs)

    alpha0 = params_file_data['alpha0']
    k = params_file_data['k']

    M1 = compute_M1_dist()

    # Krylov method
    k_krylov = np.min([k + 4, vocab_size, 2 * k])
    X = np.random.randn(vocab_size, k_krylov)
    for i in range(2 * q + 1):
        Y = multiply_M2_X_dist(X)
        X = scipy.linalg.qr(Y, mode='economic')

    Y = multiply_M2_X_dist(X)
    U, s, _ = scipy.linalg.svd(Y.T.dot(Y))

    return Q.dot(U), np.sqrt(s)
