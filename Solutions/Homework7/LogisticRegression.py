from Classifier import Classifier
from Parser import get_data_set
import numpy as np
import math


class LogisticRegression(Classifier):
    def __init__(self, X_train, y_train, learn_rate=5e-4):
        self.beta = None
        self.learn_rate = learn_rate
        self.fit(np.array(X_train), y_train)

    def fit(self, X_train, y_train):
        features_len = len(X_train[0])
        self.beta = np.matrix(np.zeros(features_len), dtype=np.float64).reshape(features_len, 1)

        for i in range(1000):
            self.beta = self.beta + self.learn_rate * \
                                    self.get_integrated_error(X_train, y_train).T

            if i % 100 == 0:
                print(self.log_likelihood(X_train, y_train))

    def predict(self, X):
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

    def log_likelihood(self, X, y):
        scores = np.dot(X, self.beta)
        ll = np.sum(y * scores - np.log(1 + np.exp(scores)))
        return ll

X_train, X_test, y_train, y_test = get_data_set(1)
lr = LogisticRegression(X_train, y_train)
score = lr.score(X_test, y_test)
print('Score: {}'.format(score))