from Classifier import Classifier
import numpy as np
from scipy.interpolate import spline
import math
from matplotlib import pyplot as plt
import os


class PCA(Classifier):
    def __init__(self, k):
        self.transformation_matrix = None
        self.k = k
        self.X_mean = None

    @staticmethod
    def get_eig_values_sort_args(eig_values, k=None):
        if k is None:
            k = len(eig_values)

        return np.flip(eig_values.argsort(), 0)[:k]

    @staticmethod
    def get_eig_vec_and_val(X):
        covariance_matrix = np.cov(X.T)
        # the covariance matrix is always symmetric, we don't need to bother any further
        return np.linalg.eig(covariance_matrix)

    @staticmethod
    def get_sorted_eig_vec(X, k=None):
        eig_values, eig_vectors = PCA.get_eig_vec_and_val(X)
        sort_args = PCA.get_eig_values_sort_args(eig_values, k)

        return eig_vectors[sort_args]

    @staticmethod
    def get_sorted_eig_values(X, k=None):
        eig_values, eig_vectors = PCA.get_eig_vec_and_val(X)
        # assert(np.all(eig_values >= 0))
        sort_args = PCA.get_eig_values_sort_args(eig_values, k)

        return eig_values[sort_args]

    def fit(self, X):
        '''
        Finds the largest k eigenvalues and their corresponding eigenvectors from the
        covariance matrix of the dataset and saves the transformation matrix
        which can be used to project a data set from it's original space to
        the space defined by those eigenvectors and their corresponding eigenvalues
        '''

        self.X_mean = X.mean(0)
        X = np.copy(X) - self.X_mean
        sorted_k_eig_vectors = PCA.get_sorted_eig_vec(X, self.k)
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

        # center data
        X = np.copy(X) - self.X_mean
        return X.dot(self.transformation_matrix)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def plot_variance_for_k(X, save_plot_name=None):
        sorted_eig_values = PCA.get_sorted_eig_values(X)
        print('got sorted eig_values')
        total_variance = sorted_eig_values.sum()
        print('got total variance')
        ks = np.arange(2, len(X[0]), 1)
        print('got ks')

        print('calculating for each k')
        variance_diffs = np.vectorize(lambda k: abs(total_variance - sorted_eig_values[:k].sum()))(ks)
        # for k in ks:
        #     variance_diffs.append(abs(total_variance - sorted_eig_values[:k].sum()))

        plt.plot(ks, variance_diffs)

        if save_plot_name is not None:
            plt.savefig(save_plot_name + '.png')
        else:
            plt.show()

