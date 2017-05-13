''' Vectorize bag-of-words dump and save in NumPy file '''
from pathlib import Path
import gzip
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bow_to_npy(vocabulary_fname, bow_fname, npy_fname, num_words=None):
    ''' Vectorize bag-of-words dump and save in NumPy file

    PARAMETERS
    -----------
    vocabulary_fname: str or Path
        Path for the vocabulary file, with one word on each line,
        in descending order of Document Frequency of the words.
    bow_fname: str or Path
        Path for the bag-of-words .txt.gz file. When uncompressed,
        each line represents a document with only lower-case words
        separated by space.
    npy_fname: str or Path
        Path for NumPy .npy file to write the word count vectors into.
    num_words: None or int
        Use all words in the vocabulary when None; otherwise only
        the top num_words words in terms of Document Frequency.
    '''
    with Path(vocabulary_fname).open('r') as vocabulary_file:
        vocabulary = [line.strip() for line in vocabulary_file]
    if num_words is not None:
        vocabulary = vocabulary[:num_words]

    vectorizer = CountVectorizer('file', vocabulary=vocabulary)
    word_counts = vectorizer.transform(gzip.open(bow_fname, 'rt'))

    np.save(npy_fname, word_counts)
