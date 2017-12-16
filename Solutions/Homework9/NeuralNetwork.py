from Classifier import Classifier
from DataNormalizer import DataNormalizer
import numpy as np
from matplotlib import pyplot as plt
from Parser import get_data_set


class NeuralNetwork(Classifier):
    def __init__(self, layers=None, max_iterations=3000, learning_rate=0.0001, online=False):
        self.data_normalizer = DataNormalizer()
        self.W_ext = None
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.online = online
        self.layers= layers

        if self.layers is None:
            self.k_layers = 2
        else:
            self.k_layers = len(self.layers)

    def fit(self, X, y):
        X = self.data_normalizer.fit_transform(X)

        if self.layers is not None:
            self.W_ext = np.array([np.ones((len(X[0]) if i == 0 else self.layers[i-1], self.layers[i]))
                                   for i in self.layers])
        else:
            dims = 30
            self.W_ext = np.array([np.ones((len(X[0]), dims)), np.ones((dims + 1, 10))])

        for i in range(self.max_iterations):
            W_ext_deltas = None
            for idx, x in enumerate(X):
                O_s = self.get_O_s(x)
                D_s = [np.diag((o * (1-o)).flatten()) for o in O_s[1:]]
                e = (O_s[-1] - NeuralNetwork.get_single_label_expectation(y[idx])).flatten()
                der_e_s = self.get_der_e_s(D_s, e)
                delta_W_ext = self.get_delta_W_ext(der_e_s, O_s)

                if W_ext_deltas is None:
                    W_ext_deltas = delta_W_ext
                else:
                    for i, delta in enumerate(delta_W_ext):
                        W_ext_deltas[i] = W_ext_deltas[i] + delta

            for i, delta in enumerate(W_ext_deltas):
                self.W_ext[i] += delta

            print(self.score(X_train, y_train))
            # print(self.W_ext[0])


    @staticmethod
    def get_label_expectation(y):
        return np.array([NeuralNetwork.get_single_label_expectation(yi) for yi in y])

    @staticmethod
    def get_single_label_expectation(y):
        y_expected = np.zeros(10)
        y_expected[int(y)] = 1
        return y_expected

    @staticmethod
    def add_ones(X):
        return np.c_[np.ones(len(X)), X]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_O_s(self, x):
        O_s = [x[1:].reshape(1, x.shape[0]-1)]
        for i in range(self.k_layers):
            O_i_minus_1 = np.c_[np.array([1]), O_s[-1]] # extend to get o(i-1)^
            O_i = NeuralNetwork.sigmoid(O_i_minus_1.dot(self.W_ext[i]))
            O_s.append(O_i)

        return O_s

    def get_der_e_s(self, D_s, e):
        W = [w[1:] for w in self.W_ext]
        der_e_l = D_s[-1].dot(e)
        der_e_l = der_e_l.reshape(der_e_l.shape[0], 1)
        der_e_s = [der_e_l]

        for i in range(self.k_layers - 1):
            der_e_i =  D_s[-i - 2].dot(W[-i - 1]).dot(der_e_s[0])
            der_e_i = der_e_i.reshape(der_e_i.shape[0], 1)
            der_e_s = [der_e_i] + der_e_s

        return der_e_s

    def get_delta_W_ext(self, der_e_s, O_s):
        delta_W_ext = []

        for i in range(self.k_layers):
            delta_W_i = (-der_e_s[i].dot(O_s[i])).T
            delta_W_i = np.vstack((delta_W_i, np.ones(delta_W_i.shape[1])))
            delta_W_ext.append(delta_W_i)

        return np.array(delta_W_ext) * self.learning_rate

    def predict(self, X):
        X = self.data_normalizer.transform(X)
        return self.predict_(X)

    def predict_(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        O_s = self.get_O_s(x)
        results = O_s[-1]
        return results.argmax()


X_train, X_test, y_train, y_test = get_data_set('digits.data')
nn = NeuralNetwork()
nn.fit(X_train, y_train)
print(nn.score)