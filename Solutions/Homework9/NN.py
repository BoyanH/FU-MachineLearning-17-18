from Classifier import Classifier
from DataNormalizer import DataNormalizer
import numpy as np
from matplotlib import pyplot as plt
import math


class NN(Classifier):
    def __init__(self, max_iterations=3000, learning_rate=0.0020,
                 size_hidden=20, size_output=10, print_score_per_bach=False, batch_size=7000):
        self.data_normalizer = DataNormalizer()
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.print_score_per_bach = print_score_per_bach
        self.W1 = None
        self.W2 = None
        self.unique_labels = None
        self.batch_size = batch_size

    def fit(self, X, y):
        self.history = []
        self.unique_labels = np.unique(y)
        X_ = self.data_normalizer.fit_transform(X)
        y_ = self.transform_y(y)
        self.W1 = np.vstack((
            np.random.randn(len(X_[0]), self.size_hidden),
            np.ones(self.size_hidden)))
        self.W2 = np.vstack((
            np.random.randn(self.size_hidden, self.size_output),
            np.ones(self.size_output)))

        batch_size = self.batch_size
        print('#Batches: {}'.format(math.ceil(len(X_) / batch_size)))
        print('#Iterations per batch: {}'.format(self.max_iterations))
        for batch_start in range(0, len(X_), batch_size):
            Xb = X_[batch_start:batch_start + batch_size]
            yb = y_[batch_start:batch_start + batch_size]
            for i in range(self.max_iterations):
                o_, o1, o1_, o2, o2_ = self.feed_forward(Xb)
                W2_ = self.W2[:-1]
                d1 = NN.sigmoid_derived(o1) # not diagonal matrix as in lecture, because sigmoid_derived(o1) is a vector
                d2 = NN.sigmoid_derived(o2)
                e = o2 - yb
                delta2 = d2 * e
                # transposing of the weights matrix missing in formula in lecture/tutorial of professor
                delta1 = d1 * (delta2.dot(W2_.T))
                deltaW2 = (-self.learning_rate * (delta2.T.dot(o1_))).T
                deltaW1 = (-self.learning_rate * delta1.T.dot(o_)).T
                self.W1 += deltaW1
                self.W2 += deltaW2

                # self.learning_rate = self.learning_rate * 1 / (1 + 0.0001 * i)

                if self.print_score_per_bach:
                    self.history.append(np.mean(self.unique_labels[o2.argmax(1)] == self.unique_labels[yb.argmax(1)]))
                else:
                    self.history.append(self.score(X, y))

            print('Batch #{} completed!'.format(math.floor(batch_start / batch_size) + 1))

    def feed_forward(self, X):
        o_ = np.c_[X, np.ones(len(X))]
        o1 = NN.sigmoid(o_.dot(self.W1))
        o1_ = np.c_[o1, np.ones(len(o1))]
        o2 = NN.sigmoid(o1_.dot(self.W2))
        o2_ = np.c_[o2, np.ones(len(o2))]

        return o_, o1, o1_, o2, o2_

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derived(x):
        return x*(1-x)

    def transform_y(self, y):
        y_ = np.zeros((len(y), len(self.unique_labels)))
        y_in_unique = np.vectorize(lambda x: list(self.unique_labels).index(x))(y)
        y_[range(len(y)), y_in_unique] = 1
        return y_

    def predict(self, X):
        X = self.data_normalizer.transform(X)
        return self.predict_(X)

    def predict_(self, X):
        o2 = self.feed_forward(X)[3]
        return self.unique_labels[o2.argmax(1)]

    def plot_accuracies(self, file_name=None):
        plt.figure(figsize=(12, 7))
        plt.plot(self.history)
        plt.xlabel("Iterations")
        plt.ylabel("accuracy")

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
