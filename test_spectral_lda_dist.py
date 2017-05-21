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
    with open(str(hostfile)) as fhost:
        for line in fhost:
            if line.startswith('#'):
                continue
            hosts.append(line.strip())

    return hosts

def test_spectral_lda_dist():
    ''' Simple test cases '''
    assert 'MXNET' in os.environ, 'MXNET must be defined'
    assert 'HOSTFILE' in os.environ, 'HOSTFILE must be defined'
    hostfile = Path(os.environ['HOSTFILE']).resolve()
    hosts = read_hostfile(hostfile)
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
    docs_path = str(Path(os.environ['HOME']) / 'test_lda')
    num_segments = 100
    partition_size = 50
    psave(docs_path, arr_iterable(docs, num_segments), docs.shape,
          partition_size=partition_size, force=True)

    # Rsync the documents to all hosts
    for host in hosts:
        run(['rsync', '-a', docs_path + '/',
             '{}:{}'.format(host, docs_path)])

    # Prepare the cmd to call
    spectral_lda_dist_file = (Path(__file__).resolve().parent /
                              'spectral_lda_dist.py')
    k = gen_k
    alpha0 = np.sum(gen_alpha[:k])
    output_prefix = str(Path(os.environ['HOME']) / 'test_lda_results')

    num_servers = 2
    num_workers = len(hosts) - num_servers

    print('hostfile: ', hostfile)
    print('docs_path: ', docs_path)
    print('spectral_lda_dist_file: ', spectral_lda_dist_file)
    print('output_prefix: ', output_prefix)

    run(['python3', 'tools/launch.py',
         '-n', str(num_workers), '-s', str(num_servers),
         '-H', str(hostfile),
         'python3', str(spectral_lda_dist_file),
         docs_path, '{:.6f}'.format(alpha0), str(k), output_prefix],
        cwd=os.environ['MXNET'])

    # Copy back the results
    run(['rm', '-f', output_prefix + '_alpha.csv'])
    run(['rm', '-f', output_prefix + '_beta.csv'])
    for host in hosts:
        run(['scp', '{}:{}_*.csv'.format(host, output_prefix),
             str(Path(output_prefix).parent)])
        if Path(output_prefix + '_alpha.csv').exists():
            break

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

def test_spectral_lda_dist_sparse():
    ''' Simple test cases '''
    assert 'MXNET' in os.environ, 'MXNET must be defined'
    assert 'HOSTFILE' in os.environ, 'HOSTFILE must be defined'
    hostfile = Path(os.environ['HOSTFILE']).resolve()
    hosts = read_hostfile(hostfile)
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
    docs = csr_matrix(docs)

    # Save the documents as partitioned arrays
    docs_path = str(Path(os.environ['HOME']) / 'test_lda')
    num_segments = 100
    partition_size = 50
    psave(docs_path, arr_iterable(docs, num_segments), docs.shape,
          partition_size=partition_size, force=True)

    # Rsync the documents to all hosts
    for host in hosts:
        run(['rsync', '-a', docs_path + '/',
             '{}:{}'.format(host, docs_path)])

    # Prepare the cmd to call
    spectral_lda_dist_file = (Path(__file__).resolve().parent /
                              'spectral_lda_dist.py')
    k = gen_k
    alpha0 = np.sum(gen_alpha[:k])
    output_prefix = str(Path(os.environ['HOME']) / 'test_lda_results')

    num_servers = 2
    num_workers = len(hosts) - num_servers

    print('hostfile: ', hostfile)
    print('docs_path: ', docs_path)
    print('spectral_lda_dist_file: ', spectral_lda_dist_file)
    print('output_prefix: ', output_prefix)

    run(['python3', 'tools/launch.py',
         '-n', str(num_workers), '-s', str(num_servers),
         '-H', str(hostfile),
         'python3', str(spectral_lda_dist_file),
         docs_path, '{:.6f}'.format(alpha0), str(k), output_prefix],
        cwd=os.environ['MXNET'])

    # Copy back the results
    run(['rm', '-f', output_prefix + '_alpha.csv'])
    run(['rm', '-f', output_prefix + '_beta.csv'])
    for host in hosts:
        run(['scp', '{}:{}_*.csv'.format(host, output_prefix),
             str(Path(output_prefix).parent)])
        if Path(output_prefix + '_alpha.csv').exists():
            break

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
