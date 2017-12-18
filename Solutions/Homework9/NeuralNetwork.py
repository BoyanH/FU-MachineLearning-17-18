from Classifier import Classifier
from DataNormalizer import DataNormalizer
import numpy as np
from matplotlib import pyplot as plt
from Parser import get_data_set


class NeuralNetwork(Classifier):
    def __init__(self, layers=None, max_iterations=3000, learning_rate=0.0025):
        """

        :param layers: tuple defining the amount of neurons to be used pro layers, thereby defining the network
        :param max_iterations:
        :param learning_rate:
        """
        self.data_normalizer = DataNormalizer()
        self.W_ext = None
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.layers = layers
        self.history = []

        if self.layers is None:
            self.k_layers = 3
        else:
            self.k_layers = len(self.layers)

    def transform_y(self, y):
        y_ = np.zeros((len(y), len(self.unique_labels)))
        y_in_unique = np.vectorize(lambda x: list(self.unique_labels).index(x))(y)
        y_[range(len(y)), y_in_unique] = 1
        return y_

    def fit(self, X, y):
        self.unique_labels = np.unique(y)
        X_ = self.data_normalizer.fit_transform(X)
        y_ = self.transform_y(y)

        if self.layers is not None:
            self.W_ext = np.array([np.random.randn(len(X_[0] + 1, self.layers[i])) if i == 0 else
                                   np.random.randn(self.layers[i - 1] + 1, self.layers[i])
                                   for i in self.layers])
        else:
            self.W_ext = np.array([np.random.randn(len(X_[0]) + 1, 20), np.random.randn(20 + 1, 15),
                                   np.random.randn(15+1, 10)])

        batch_size = 1000
        for batch_start in range(0, len(X_), batch_size):
            Xb = X_[batch_start:batch_start + batch_size + 1]
            yb = y_[batch_start:batch_start + batch_size + 1]
            for it in range(self.max_iterations):
                O_s = self.get_O_s(Xb)
                D_s = [(o * (1.0 - o)) for o in O_s[2::2]]
                e = (O_s[-1] - yb)
                der_e_s = self.get_der_e_s(D_s, e)
                delta_W_ext = self.get_delta_W_ext(der_e_s, O_s[::2])

                for i, delta in enumerate(delta_W_ext):
                    self.W_ext[i] += delta

                self.history.append(self.score(X, y))

    @staticmethod
    def add_ones(X):
        return np.c_[np.ones(len(X)), X]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_O_s(self, X):
        O_s = [X]
        for i in range(self.k_layers):
            O_i_minus_1 = np.c_[O_s[-1], np.ones(len(X))]  # extend to get o(i-1)^
            O_s.append(O_i_minus_1)
            O_i = NeuralNetwork.sigmoid(O_i_minus_1.dot(self.W_ext[i]))
            O_s.append(O_i)

        return O_s

    def get_der_e_s(self, D_s, e):
        W = [w[1:] for w in self.W_ext]
        der_e_l = e * D_s[-1]
        der_e_s = [der_e_l]

        for i in range(self.k_layers - 1):
            der_e_i = D_s[-i - 2] * (der_e_s[0]).dot(W[-i - 1].T)
            der_e_s = [der_e_i] + der_e_s

        return der_e_s

    def get_delta_W_ext(self, der_e_s, O_s):
        delta_W_ext = []

        for i in range(self.k_layers):
            o_i_ext = np.c_[O_s[i], np.ones(len(O_s[i]))]
            delta_W_i = der_e_s[i].T.dot(o_i_ext)
            delta_W_ext.append(delta_W_i.T)

        return np.array(delta_W_ext) * -self.learning_rate

    def predict(self, X):
        X = self.data_normalizer.transform(X)
        return self.predict_(X)

    def predict_(self, X):
        O_s = self.get_O_s(X)
        results = O_s[-1]
        return self.unique_labels[results.argmax(1)]

    def plot_accuracies(self, file_name=None):
        plt.figure(figsize=(12, 7))
        plt.plot(self.history)
        plt.xlabel("Iterations")
        plt.ylabel("accuracy")

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)


X_train, X_test, y_train, y_test = get_data_set('digits.data')
nn = NeuralNetwork()
nn.fit(X_train, y_train)
nn.plot_accuracies('./2_hidden_layers_20_15_with_7_batches.png')
print('Score train: {}'.format(nn.score(X_train, y_train)))
print('Score: {}'.format(nn.score(X_test, y_test)))
