''' Test partitioned data '''

from random import sample
import os
from tempfile import gettempdir
from subprocess import run
from partitioned_data import pload, psave
import numpy as np
import scipy.sparse as sps

def arr_iterable(arr, num_segments):
    ''' Simulate arr iterable '''
    height, width = arr.shape
    assert height >= 1 and width >= 1
    assert num_segments >= 1
    segments = ([0] + sorted(sample(range(1, height), num_segments - 1))
                + [height])
    for start, end in zip(segments[:-1], segments[1:]):
        yield arr[start:end, :]

def test_partitioned_data():
    ''' Simple test cases '''
    # Simulate the ground-truth arr
    height, width = 100, 10000
    arr = np.random.rand(height, width)

    # Save arr as partitioned_data
    fname = os.path.join(gettempdir(), 'test_pdata')
    num_segments = 50
    partition_size = 20
    psave(fname, arr_iterable(arr, num_segments), arr.shape,
          partition_size=partition_size, force=True)

    # Test reading of whole file
    arr_loaded = pload(fname, 0, height)
    assert np.allclose(arr, arr_loaded)

    # Test random access
    for _ in range(50):
        start, end = sorted(sample(range(height + 1), 2))
        assert np.allclose(arr[start:end, :], pload(fname, start, end))

    run(['rm -rf ' + fname], shell=True)

def test_partitioned_data_sparse():
    ''' Simple test cases '''
    # Simulate the ground-truth arr
    height, width = 100, 10000
    arr = np.random.rand(height, width)
    arr = sps.csr_matrix(arr)

    # Save arr as partitioned_data
    fname = os.path.join(gettempdir(), 'test_pdata_sps')
    num_segments = 50
    partition_size = 20
    psave(fname, arr_iterable(arr, num_segments), arr.shape,
          partition_size=partition_size, force=True)

    # Test reading of whole file
    arr_loaded = pload(fname, 0, height)
    print('Expected\n', arr[:5, :].toarray())
    print('Obtained\n', arr_loaded[:5, :].toarray())
    assert (arr != arr_loaded).nnz == 0

    # Test random access
    for _ in range(50):
        start, end = sorted(sample(range(height + 1), 2))
        assert (arr[start:end, :] != pload(fname, start, end)).nnz == 0

    run(['rm -rf ' + fname], shell=True)
