from Classifier import Classifier
import numpy as np
import math
from matplotlib import pyplot as plt
import os


class PCA(Classifier):
    def __init__(self, k):
        self.transformation_matrix = None
        self.k = k

    def fit(self, X):
        '''
        Finds the largest k eigenvalues and their corresponding eigenvectors from the
        covariance matrix of the dataset and saves the transformation matrix
        which can be used to project a data set from it's original space to
        the space defined by those eigenvectors and their corresponding eigenvalues
        '''

        covariance_matrix = np.cov(X.T)
        # as the covariance matrix is always symmetric, we can use eigh for more correctness
        eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

        sorted_k_eig_values_args = np.flip(eig_values.argsort(), 0)[:self.k]  # sort descending
        sorted_k_eig_vectors = eig_vectors[sorted_k_eig_values_args]

        self.transformation_matrix = sorted_k_eig_vectors.T

    def transform(self, X):
        '''
        (x11 x12 ... x1n )             (e11 e12 ... e1n)T       (x1 projected in subspace)
        (... ... ... ... )      X      (... ... ... ...)    =   (...                     )
        (xm1 xm2 ... xmn )             (ek1 ek2 ... ekn)        (xm projected in subspace)

        self.transformation_matrix is the already transponded matrix where each row is an eigenvector

        :param X: data set, each row represents a sample
        :return: X in the space defined by the first k eigenvectors of the fit data set
        '''

        return X.dot(self.transformation_matrix)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


