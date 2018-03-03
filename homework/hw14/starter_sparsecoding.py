import sklearn.decomposition
import numpy as np
def sparse_coding(D,X,s):  
    '''
    This function implements sparse coding in the pseudo code.
    ''' 
    Z = sklearn.decomposition.sparse_encode(X,
                                            D,
                                            algorithm='omp',
                                            alpha = 1.,
                                            n_nonzero_coefs=s,
                                           max_iter = 100) 
    
    return Z


def compute_error(X,D,Z):
    """
    Compute reconstruction MSE. 
    """

    error = np.linalg.norm(X - Z.dot(D), ord='fro')**2/len(X)
    return error

def generate_Z(N,K,s_true):
    """
    Generate random coefficient matrix. 
    """
    Z = np.zeros((N,K))
    zero_one_vec = np.zeros(K).astype(int)
    zero_one_vec[:s_true] = 1

    sparse_indicator = np.array([np.random.permutation(zero_one_vec) for i in range(N)])
    Z[np.where(sparse_indicator)]=1.0 
    return Z


def generate_test_data(N, s, K, d):
    """
    Generate a dataset for the testing purpose. 
    """
    Z = generate_Z(N,K,s)
    D = np.random.randn(K,d) 
    X = Z.dot(D)
    return X   

def generate_toy_data(N, s, K, d, c):
    def f(j):
        return 10/j
    sampling_loc = np.expand_dims(np.arange(0,1,1.0/d),0)
    freq = np.expand_dims(np.arange(1,K+1),1)
    
    D = np.sin(2*np.pi*freq.dot(sampling_loc))
    D = D / np.expand_dims(np.sqrt(np.sum(D**2, axis = 1)),1)
    Z = generate_Z(N,K,s)

    X = Z.dot(D) + c * np.random.randn(N,d)
    return X,D,Z