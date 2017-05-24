# SpectralLDA-MXNet
Spectral LDA on MXNet

## Summary 
This code implements a Spectral (third order tensor decomposition) learning method for learning LDA topic model on MXNet.

The Spectral learning method works with empirical counts of word pair or word triplet that appear in the same document. We collect and average them across documents. If we denote these empirical moments as tensors, we could orthogonalise and then perform the CANDECOMP/PARAFAC Decomposition on the 3rd-order moment tensor to recover the topic-word distributions of the LDA model. For more details, please refer to `report.pdf` in this repository.

## Usage
We need to first set up MXNet by following its instructions:

<https://github.com/dmlc/mxnet>

Compile the CPDecomp Op that's currently still in Pull Request:

<https://github.com/dmlc/mxnet/pull/5486>

To run the code in distributed mode, make sure we have rsynced the data files on all the worker nodes, 

```bash
PS_VERBOSE=1 python3 <mxnet>/tools/launch.py \
-n <num_workers> -s <num_servers> -H <hostfile> \
python3 spectral_lda_dist.py <data_path> <alpha0> <k> <output_prefix>
```

### Prepare Text Corpus Data
We could use the `sklearn.feature_extraction.text.CountVectorizer` in `scikit-learn` to convert the documents into sparse matrices of word counts. Denote `arr` as an iterable over the sparse matrices of word counts (the number of rows of each matrix could be irregular), we provided `partitioned_data.py` to save the corpus into partitioned files.

```python
from partitioned_data import psave
psave(fname, arr, shape, partition_size, force=True)
```

where `fname` is the path to save the partitioned files in, `shape` is the shape of the *entire* corpus word count matrix, `partition_size` indicates the number of rows saved in each file, `force` if set to `True` will override existing files under `fname`.

Once saved, `fname` could be used as `data_path` for the command line usage above.

## References
* White Paper: http://newport.eecs.uci.edu/anandkumar/pubs/whitepaper.pdf
* New York Times Result Visualization: http://newport.eecs.uci.edu/anandkumar/Lab/Lab_sub/NewYorkTimes3.html



