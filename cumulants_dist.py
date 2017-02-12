import mxnet as mx
import numpy as np
import argparse
import os

from cumulants import *


def docs_range(n_docs):
    range_size = n_docs // os.environ['DMLC_NUM_WORKER']
    worker_id = os.environ['DMLC_WORKER_ID']
    return range(worker_id * range_size, 
                 min((worker_id + 1) * range_size, n_docs))

def compute_M1_dist(docs_file, params_file):
    kvstore = mx.kvstore.create('dist_sync')
    
    docs_file_data = np.load(docs_file)
    params_file_data = np.load(params_file)

    docs = docs_file_data['docs']
    vocab_size = docs_file_data['vocab_size']
    n_docs = len(docs)

    alpha0 = params_file_data['alpha0']
    k = params_file_data['k']

    result_key = 90 
    if 'DMLC_WORKER_ID' in os.environ:
        result = mx.nd.zeros((vocab_size,))
        kvstore.init(result_key, result)
        
        m1 = docs[docs_range(n_docs), :].mean(axis=0)
        kvstore.push(result_key, m1)
        
        kvstore.pull(result_key, out=result)
    
        np.savez(params_file, 
                 M1=result.asnumpy(),
                 alpha0=alpha0,
                 k=k)

    kvstore.close()

def mult_M2_X_dist(docs_file, params_file):
    kvstore = mx.kvstore.create('dist_sync')
    
    docs_file_data = np.load(docs_file)
    params_file_data = np.load(params_file)

    docs = docs_file_data['docs']
    vocab_size = docs_file_data['vocab_size']
    n_docs = len(docs)

    X = params_file_data['X']
    M1 = params_file_data['M1']
    alpha0 = params_file_data['alpha0']
    k = params_file_data['k']
    
    result_key = 100
    if 'DMLC_WORKER_ID' in os.environ:
        result = mx.nd.zeros((vocab_size, k))
        kvstore.init(result_key, result)
        
        prod_E2_X = mult_E2_X(docs[docs_range(n_docs), :], X, n_docs)
        kvstore.push(result_key, prod_E2_X)
        
        kvstore.pull(result_key, out=result)
    
        prod_M2_X = mult_M2_X_helper(result.asnumpy(), X, M1, alpha0)
        np.savez(params_file, 
                 prod_M2_X=prod_M2_X,
                 X=X,
                 M1=M1,
                 alpha0=alpha0,
                 k=k)

    kvstore.close()


def whiten_M3_dist(docs_file, params_file):
    kvstore = mx.kvstore.create('dist_sync')
    
    docs_file_data = np.load(docs_file)
    params_file_data = np.load(params_file)

    docs = docs_file_data['docs']
    vocab_size = docs_file_data['vocab_size']
    n_docs = len(docs)

    W = params_file_data['W']
    M1 = params_file_data['M1']
    alpha0 = params_file_data['alpha0']
    k = params_file_data['k']
    
    whitened_E3_key = 101 
    whitened_E2_M1_key = 102
    if 'DMLC_WORKER_ID' in os.environ:
        whitened_E3_mx = mx.nd.zeros((k, k, k))
        whitened_E2_M1_mx = mx.nd.zeros((k, k, k))
        kvstore.init(whitened_E3_key, whitened_E3_mx)
        kvstore.init(whitened_E2_M1_key, whitened_E2_M1_mx)

        whitened_E3 = whiten_E3(docs[docs_range(n_docs), :], X, n_docs)
        whitened_E2_M1 = whiten_E2_M1(docs[docs_range(n_docs), :], X, n_docs, M1)
        kvstore.push(whitened_E3_key, whitened_E3)
        kvstore.push(whitened_E2_M1_key, whitened_E2_M1)
        
        kvstore.pull(whitened_E3_key, out=whitened_E3_mx)
        kvstore.pull(whitened_E2_M1_key, out=whitened_E2_M1_mx)
    
        whitened_M3 = whiten_M3_helper(whitened_E3_mx.asnumpy(),
                                       whitened_E2_M1_mx.asnumpy(), 
                                       W, M1, alpha0)
        np.savez(params_file, 
                 whitened_M3=whitened_M3,
                 W=W,
                 M1=M1,
                 alpha0=alpha0,
                 k=k)

    kvstore.close()


if __name__ = '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('cmd', 
                        choices=['compute_M1', 'mult_M2_X', 'whiten_M3'],
                        help='command')
    parser.add_argument('--data-path',
                        default='/tmp/speclda/',
                        help='path for all data files. default: /tmp/speclda/')
    parser.add_argument('--docs',
                        default='docs.npz',
                        help='document word counts file. default: docs.npz')
    parser.add_argument('--params',
                        default='params.npz',
                        help='all the other params file. default: params.npz')

    args = parser.parse_args()
    docs_file = os.path.join(args.data_path, args.docs)
    params_file = os.path.join(args.data_path, args.params)

    if args.cmd == 'compute_M1':
        compute_M1(docs_file, params_file)
    elif args.cmd == 'mult_M2_X':
        mult_M2_X_dist(docs_file, params_file)
    elif args.cmd == 'whiten_M3':
        whiten_M3_dist(docs_file, params_file)




