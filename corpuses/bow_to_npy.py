''' Vectorize bag-of-words dump and save in NumPy file '''
from pathlib import Path
import gzip
from argparse import ArgumentParser
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bow_to_npy(vocabulary_fname, bow_fname, npy_fname):
    ''' Vectorize bag-of-words dump and save in NumPy file

    PARAMETERS
    -----------
    vocabulary_fname: str or Path
        Vocabulary text file name, with one word on each line.
    bow_fname: str or Path
        Bag-of-words .txt.gz file name. When uncompressed,
        each line represents a document with only lower-case words
        separated by space.
    npy_fname: str or Path
        NumPy .npy file name to write the word count vectors into.
    '''
    with Path(vocabulary_fname).open('r') as vocabulary_file:
        vocabulary = [line.strip() for line in vocabulary_file]

    vectorizer = CountVectorizer(vocabulary=vocabulary)
    with gzip.open(bow_fname, 'rt') as bow_file:
        word_counts = vectorizer.transform(bow_file)

    np.save(npy_fname, word_counts)

def main():
    ''' Parse arguments and run '''
    parser = ArgumentParser(description='Vectorize bag-of-words dump files')

    parser.add_argument('vocabulary_fname', type=str,
                        help='Vocabulary text file name')
    parser.add_argument('bow_fname', type=str,
                        help='Bag-of-words .txt.gz file name')
    parser.add_argument('npy_fname', type=str,
                        help='.npy file name for saving the word counts')

    args = parser.parse_args()
    bow_to_npy(**vars(args))

if __name__ == '__main__':
    main()
