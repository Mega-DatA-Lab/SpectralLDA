''' Dump Bag-of-Word from gensim WikiCorpus '''
import gzip
from argparse import ArgumentParser
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.wikicorpus import WikiCorpus

def dump_bow(corpus, partition_size=50, limit=200, output_prefix='dump'):
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
    limit: int or None
        The total number of documents to dump, or all the documents
        when it is None.
    output_prefix: str
        Prefix of the dump files.
    '''
    def write_buffer(buf, output_prefix, partition_id):
        ''' Dump current buffer of Bag-of-Words '''
        fname = '{}-{:06d}.txt.gz'.format(output_prefix, partition_id)
        with gzip.open(fname, 'wt') as partition_file:
            partition_file.write(buf)

    if limit is None:
        limit = len(corpus)

    assert partition_size >= 1 and limit >= 1
    # gensim 2.0 requires this otherwise the multi-processing locks up
    assert partition_size <= limit

    count_documents = 0
    partition_id = 0
    buf = ''

    for bow in corpus.get_texts():
        text = ' '.join([byte_array.decode('utf-8') for byte_array in bow])
        buf += text + '\n'
        count_documents += 1

        if count_documents % 200 == 0:
            print('Processes {} documents.'.format(count_documents))

        if count_documents % partition_size == 0:
            write_buffer(buf, output_prefix, partition_id)
            buf = ''
            partition_id += 1

        if count_documents >= limit:
            break

    if buf:
        write_buffer(buf, output_prefix, partition_id)

def main():
    ''' Parse arguments and run '''
    parser = ArgumentParser(description='Dump bag-of-words in .txt.gz files')

    parser.add_argument('wikidump', type=str,
                        help='xxx-pages-articles.xml.bz2 wiki dump file')
    parser.add_argument('dictionary', type=str,
                        help='gensim dictionary .txt file')

    parser.add_argument('-j', '--jobs', type=int, default=2,
                        help='Number of parallel jobs, default: 2')
    parser.add_argument('-p', '--partition-size', type=int, default=50,
                        help=('Number of documents in each .txt.gz file, '
                              'default: 50'))
    parser.add_argument('-l', '--limit', type=int, default=200,
                        help=('Total number of documents to dump, '
                              'or -1 for all documents, default: 200'))
    parser.add_argument('-o', '--output-prefix', type=str, default='dump',
                        help='Prefix of dump .txt.gz files, default: dump')

    args = parser.parse_args()

    wiki_dictionary = Dictionary.load_from_text(args.dictionary)
    wiki = WikiCorpus(args.wikidump, processes=args.jobs,
                      dictionary=wiki_dictionary)

    if args.limit == -1:
        dump_bow(wiki, args.partition_size, None,
                 output_prefix=args.output_prefix)
        print('Dumped {} documents.'.format(len(wiki)))
    else:
        dump_bow(wiki, args.partition_size, args.limit,
                 output_prefix=args.output_prefix)
        print('Dumped {} documents.'.format(args.limit))

if __name__ == '__main__':
    main()
