from Classifier import Classifier
import numpy as np
import math
from matplotlib import pyplot as plt
import os


class LogisticRegression(Classifier):
    # def __init__(self, X_train, y_train, learn_rate=1e-3):
    def __init__(self, X_train, y_train, learn_rate=1e-3, iterations=40000, plot=False):
        self.beta = None
        self.transformation_vector = None
        self.learn_rate = learn_rate
        X, y = self.transform(X_train, y_train)
        self.fit(X, y, iterations, plot)

    @staticmethod
    def sigmoid(weighted):
        return 1 / (1 + np.exp(-weighted))

    def transform(self, X, y):

        # feature_means = X.sum(0) / len(X[0])
        # X_t = X.T
        # variance = [((X_t[i] - feature_means[i])**2).sum() for i in range(len(X_t))]

        # numpy does it more effectively and normalized
        self.transformation_vector = np.var(X, axis=0)
        X = X / self.transformation_vector

        # add ones
        ones = np.ones((len(X), 1), dtype=np.float64)
        X = np.append(ones, X, axis=1)

        return X, y

    def fit(self, X, y, iterations, plot):
        features_len = len(X[0])
        self.beta = np.zeros(features_len, dtype=np.float64)
        log_error_over_time = []

        last_log_error = float('inf')
        current_learning_rate = self.learn_rate

        for i in range(iterations):
            weighted = X.dot(self.beta)
            probabilities = LogisticRegression.sigmoid(weighted)
            directions = y - probabilities
            gradient = X.T.dot(directions)

            # Bold driver technique
            # If error was actually larger (overshooting) use previous weight vector
            # and decrease learning rate by 50%; otherwise increase learn rate by 5%

            # self.beta = self.beta + (self.learn_rate / (2**(i/5000)))*gradient
            self.learn_rate = self.learn_rate / 2
            self.beta = self.beta + current_learning_rate*gradient

            if i % 100 == 0:
                current_log_error = abs(self.get_log_likelihood(X, y))

                if current_log_error < last_log_error:
                    current_learning_rate += current_learning_rate * 0.0003  # increase by 0.03%
                else:
                    self.beta = last_beta
                    current_learning_rate -= current_learning_rate * 0.03 # decrease by 3%

                last_log_error = current_log_error
                last_beta = np.copy(self.beta)

                if plot:
                    log_error_over_time.append(current_log_error)

        if plot:
            plt.ylabel('Absolute Log Likelihood')
            plt.xlabel('Iterations')
            plt.plot([i for i in range(0, iterations, 100)], log_error_over_time)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'll_over_time.png'))

    def predict(self, X):
        X = X / self.transformation_vector
        # add ones
        ones = np.ones((len(X), 1), dtype=np.float64)
        X = np.append(ones, X, axis=1)
        return list(map(lambda x: self.predict_single(x), X))

    def get_integrated_error(self, X, y):

        data_len = len(y)
        get_integrated_error_per_data_point = np.vectorize(
            lambda idx: y[idx]*X[idx]*(1 - self.get_probability(X[idx], y[idx])),
            signature='()->(m)')
        integrated_errors = get_integrated_error_per_data_point(range(data_len))

        return integrated_errors.sum(0)

    def get_probability(self, x, y):
        try:
            return 1 / (1 + math.exp((-y * self.beta).T.dot(x)))
        except:
            return 0

    def predict_single(self, x):
        return 1 if self.get_probability(x, 1) > self.get_probability(x, 0) else 0

    def get_log_likelihood(self, X, y):
        weighted = X.dot(self.beta)
        return np.sum(y * weighted - np.log(1 + np.exp(weighted)))
