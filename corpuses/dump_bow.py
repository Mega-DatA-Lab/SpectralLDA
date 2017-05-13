''' Dump Bag-of-Word from gensim WikiCorpus '''
import gzip

def dump_bow(corpus, partition_size=10000, limit=200, output_prefix='dump'):
    ''' Dump Bag-of-Word from gensim WikiCorpus

    Iterate through the documents in the wiki dump and dump the
    Bag-of-Words of the documents in a series of .txt.gz files.
    Each line in the uncompressed file represent one document, with
    only lower-case words separated by space.

    PARAMETERS
    -----------
    corpus: gensim.corpora.WikiCorpus
        The Wikidump corpus.
    partition_size: int
        Number of documents in each .txt.gz dump file.
    limit: int
        The total number of documents to dump.
    output_prefix: str
        Prefix of the dump files.
    '''
    def write_buffer(buf, output_prefix, partition_id):
        ''' Dump current buffer of Bag-of-Words '''
        fname = '{}-{:06d}.txt.gz'.format(output_prefix, partition_id)
        with gzip.open(fname, 'wt') as partition_file:
            partition_file.write(buf)

    count_documents = 0
    partition_id = 0
    buf = ''

    for bow in corpus.get_texts():
        text = ' '.join([byte_array.decode('utf-8') for byte_array in bow])
        buf += text + '\n'
        count_documents += 1

        if count_documents % partition_size == 0:
            write_buffer(buf, output_prefix, partition_id)
            buf = ''
            partition_id += 1

        if limit is not None and count_documents >= limit:
            return

    if buf:
        write_buffer(buf, output_prefix, partition_id)
