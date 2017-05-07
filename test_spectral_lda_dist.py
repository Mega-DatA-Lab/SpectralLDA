''' Test Spectral LDA in distributed mode

The environment variable MXNET needs to be set so that we
could call ${MXNET}/tools/launch.py. HOSTFILE needs be set
as well to the filepath of the Hostfile.

We use 2 servers and the rest nodes as workers for test. Therefore
there has to be at least 4 nodes in Hostfile.
'''
import os
from pathlib import Path
from subprocess import run
import numpy as np
from scipy.sparse import csr_matrix
from partitioned_data import psave
from test_partitioned_data import arr_iterable
from test_cumulants import simulate_word_count_vectors

def read_hostfile(hostfile):
    ''' Return hosts '''
    hosts = []
    with open(hostfile) as fhost:
        for line in fhost:
            if line.startswith('#'):
                continue
            hosts.append(line.strip())

    return hosts

def test_spectral_lda_dist():
    ''' Simple test cases '''
    assert 'MXNET' in os.environ, 'MXNET must be defined'
    assert 'HOSTFILE' in os.environ, 'HOSTFILE must be defined'
    hosts = read_hostfile(os.environ['HOSTFILE'])
    assert len(hosts) >= 4, 'There must be at least 4 hosts'

    # Generative alpha
    gen_alpha = [10, 5, 2]
    gen_k = len(gen_alpha)

    vocab_size = 50
    n_docs = 5000

    # Generative beta
    gen_beta = np.random.rand(vocab_size, gen_k)
    gen_beta /= gen_beta.sum(axis=0)

    # Simulate the documents
    docs = simulate_word_count_vectors(gen_alpha, gen_beta, n_docs,
                                       500, 1000)

    # Save the documents as partitioned arrays
    docs_path = '/tmp/test_lda'
    num_segments = 100
    partition_size = 50
    psave(docs_path, arr_iterable(docs, num_segments), docs.shape,
          partition_size=partition_size)

    # Prepare the cmd to call
    k = gen_k
    alpha0 = np.sum(gen_alpha[:k])
    output_prefix = '/tmp/test_lda_results'

    num_servers = 2
    num_workers = len(hosts) - num_servers
    run(['python3', 'tools/launch.py', '-n', num_workers, '-s', num_servers,
         '-H', os.environ['HOSTFILE'], '--sync-dst-dir', docs_path,
         'python3', str(Path(__file__).resolve()),
         docs_path, '{:.6f}'.format(alpha0), str(k), output_prefix],
        cwd=os.environ['MXNET'])

    # Copy back the results
    for host in hosts:
        run(['scp', '{}:{}_*.csv'.format(host, output_prefix),
             str(Path(output_prefix).parent)])

    alpha = np.loadtxt(output_prefix + '_alpha.csv')
    beta = np.loadtxt(output_prefix + '_beta.csv')

    print('Generative alpha:')
    print(gen_alpha)
    print('Fitted alpha:')
    print(alpha)

    print('Generative beta:')
    print(gen_beta)
    print('Fitted beta:')
    print(beta)

    assert np.all(np.linalg.norm(gen_beta[:, :k] - beta, axis=0) < 0.2)
