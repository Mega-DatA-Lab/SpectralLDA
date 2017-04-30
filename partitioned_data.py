''' Routines for accessing partitioned array

For a data set we could store the meta information and data partitions
under a directory. The list of files are:

    .meta
    p000000.npy
    p000001.npy
    p000002.npy
    ...

The layout of `.meta` is

    # height width partition_size
    <height> <width> <partition_size>

which gives the number of rows (`height`), columns (`width`) of the entire
data set, the number of rows of each parition (`partition_size`).

`p000000.npy`, `p000001.npy`, `p000002.npy`, etc. store the partitions of
the data, each with `partition_size` rows. Each partition could be a NumPy
array or sparse csr_matrix.
'''
from pathlib import Path
import subprocess
import numpy as np
import scipy.sparse as sps

def pmeta(fname):
    ''' Return meta data of the partitioned array

    PARAMETERS
    -----------
    fname : str
        Path to the partitioned array.

    RETURNS
    -----------
    height : int
        Number of rows of the entire data set.
    width : int
        Numer of columns of the entire data set.
    partition_size : int
        Number of rows in each partition.
    '''
    height, width, partition_size = np.loadtxt(fname + '/.meta', dtype=int)
    assert height >= 1 and width >= 1
    assert partition_size >= 1
    return height, width, partition_size

def _vstack(partitions):
    ''' vstack array or sparse matrix '''
    # pylint: disable=no-else-return
    if sps.issparse(partitions[0]):
        return sps.vstack(partitions)
    else:
        return np.vstack(partitions)

def pload(fname, start, end):
    ''' Load specified range of the partitioned array

    PARAMETERS
    -----------
    fname : str
        Path to the partitioned array.
    start : int
        Index of the starting row, inclusive.
    end : int
        Index of the ending row, exclusive.

    RETURNS
    -----------
    out : array or csr_matrix
        Specified range of the partitioned array.
    '''

    height, _, partition_size = pmeta(fname)
    assert start >= 0 and end >= start
    assert end <= height

    # Compute the start and end partition IDs
    start_partition_id = start // partition_size
    end_partition_id = (end + partition_size - 1) // partition_size
    assert end_partition_id <= 1e6

    # Retrieve all the partitions
    path = Path(fname)
    partitions = []
    for partition_id in range(start_partition_id, end_partition_id):
        partition = np.load(path / 'p{:06d}.npy'.format(partition_id))
        # Sparse csr_matrix will be read inside a size-1 ndarray
        if partition.dtype == 'object':
            partition = partition.item()

        partitions.append(partition)

    # Concatenate into a superset of the requested data
    superset = _vstack(partitions)

    # Return requested data
    offset = start_partition_id * partition_size
    return superset[(start - offset):(end - offset)]

def psave(fname, arr, shape, partition_size, force=False):
    ''' Save partitioned array

    PARAMETERS
    -----------
    fname : str
        Path under which to save the partitioned array.
    arr : iterable
        Iterable of partitions of the array, each partition
        could be NumPy array or sparse csr_matrix.
    shape : tuple
        Shape of the entire array.
    partition_size : int
        Size of each partition.
    force : bool, optional
        False by default, for which no writing is performed if
        the path is non-empty. Setting to True will force writing.
    '''
    path = Path(fname)

    # Check if path is non-empty
    # only quit if force=False
    if not force and (path / '.meta').exists():
        raise RuntimeError('%s non-empty', path)

    # Make path
    subprocess.run(['rm -rf {}'.format(path)], shell=True)
    path.mkdir()

    # Write .meta
    with (path / '.meta').open(mode='w') as fmeta:
        fmeta.write('# height width partition_size\n')
        fmeta.write('{} {} {}'.format(shape[0], shape[1],
                                      partition_size))

    # Every time we read in at least partition_size rows,
    # we write them into partition files
    count_rows = 0
    list_rows = []
    partition_id = 0
    for rows in arr:
        list_rows.append(rows)
        count_rows += rows.shape[0]

        if count_rows >= partition_size:
            cache_arr = _vstack(list_rows)
            for i in range(count_rows // partition_size):
                fpart = path / 'p{:06d}.npy'.format(partition_id + i)
                np.save(fpart, cache_arr[(i * partition_size):
                                         ((i + 1) * partition_size)])

            partition_id += count_rows // partition_size
            remaining_rows = count_rows % partition_size
            list_rows.clear()
            if remaining_rows > 0:
                list_rows.append(cache_arr[-remaining_rows:, :])
            count_rows = remaining_rows

    # Write the last partition if there is
    if count_rows > 0:
        assert count_rows < partition_size
        fpart = path / 'p{:06d}.npy'.format(partition_id)
        np.save(fpart, _vstack(list_rows))
