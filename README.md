# SpectralLDA

**Note: This is the single-host version, for the up-to-date and distributed version please refer to [https://github.com/Mega-DatA-Lab/SpectralLDA-Spark].**

This code implements a Spectral (third order tensor decomposition) learning method for the Latent Dirichlet Allocation model in Python.

The Spectral learning method works with empirical counts of word pair or word triplet from any document in the dataset. We average the counts and put them in tensors. We then perform tensor decomposition to learn the Latent Dirichlet Allocation model. For more details, please refer to `report.pdf` in the repository.

## Usage
Invoke `spectral_lda` with the doc-term count matrix. At output we'd learn `alpha` for the Dirichlet prior parameter, `beta` for the topic-word-distribution, with one topic per column.

```python
# docs is the doc-term count matrix
# alpha0 is the sum of the Dirichlet prior parameter
# k is the rank aka number of topics
from spectral_lda import spectral_lda
alpha, beta = spectral_lda(docs, alpha0=<alpha0>, k=<k>, l1_simplex_proj=False)

# alpha is the learnt Dirichlet prior
# beta is the topic-word-distribution matrix
# with one column per topic
```

By default each column in `beta` may not sum to one, set `l1_simplex_proj=True` to perform post-processing that projects `beta` into the l1-simplex.

## References
Anandkumar, Animashree, Rong Ge, Daniel Hsu, Sham M. Kakade, and Matus Telgarsky, Tensor Decompositions for Learning Latent Variable Models.
