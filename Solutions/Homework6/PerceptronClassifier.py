from Classifier import Classifier
from Parser import get_data_set, extract_classes_from_data_set
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from random import random, sample
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

classes_in_data_set = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
infinity = float('inf')
learning_rate = 0.0001

class PerceptronClassifier(Classifier):
    def __init__(self, X, y, class_a, class_b):
        ones = np.ones((len(X), 1), dtype=np.float64)
        X_normalized = np.append(ones, X, axis=1)
        y_normalized = np.vectorize(lambda x: -1 if x == class_a else 1)(y)
        np.random.seed(8)
        self.w = X_normalized[np.random.randint(0, X_normalized.shape[0], 1)][0]
        self.class_a = class_a
        self.class_b = class_b
        self.fit(X_normalized, y_normalized)

    def fit(self, X, y):
        t = 0
        least_error = None
        current_error = None
        best_w = self.w
        worse_iterations = 0

        while worse_iterations < 500 and (current_error is None or current_error > 0):  #  < least_error:
            if least_error is not None and current_error > least_error:
                worse_iterations += 1

            current_error = 0

            # what happens here is really the same as in the lecture
            # just without if statements; if e.g x is positive and we predicted negative
            # predict single would be 0, y would be 1
            # => self.w + learning_rate*x

            # learning rate is something commonly used in this algorithm, in the lecture we learned
            # a simplified version where the learning rate is 1
            for x, yi in zip(X, y):
                error = ((self.predict_single_normalized(x) - yi)/2)
                w_new = self.w - learning_rate*error*x
                current_error += PerceptronClassifier.get_error(w_new, self.w)
                self.w = w_new

            if current_error is not None and (least_error is None or current_error < least_error):
                least_error = current_error
                best_w = self.w

        for x, yi in zip(X, y):
            error = (self.predict_single_normalized(x) - yi)
            assert(error == 0 or current_error > 0)

        print('Least error for {} vs {}: {}deg'.format(self.class_a, self.class_b, least_error))

        self.w = best_w / np.linalg.norm(best_w)

    @staticmethod
    def get_error(w_new, w):
        # err in degrees rotation
        return math.acos(np.clip((w_new / np.linalg.norm(w_new)).dot(w / np.linalg.norm(w)), -1.0, 1.0))

    def project_point(self, x):
        return x.dot(self.w / np.linalg.norm(self.w))

    def predict_single_normalized(self, x):
        return 1 if self.project_point(x) > 0 else -1

    def predict_single(self, x):
        x_normalized = np.append(np.array([1]), x)
        return self.class_a if self.predict_single_normalized(x_normalized) == 0 else self.class_b

    def predict(self, X):
        return np.vectorize(lambda x: self.predict_single(x), signature='(n)->()')(X)


# Classes
# Iris-setosa, Iris-versicolour, Iris-virginica

X_train, X_test, y_train, y_test = get_data_set(1)

X_vi_se_train, y_vi_se_train = extract_classes_from_data_set(X_train, y_train, classes_in_data_set[:-1])
X_vi_se_test, y_vi_se_test = extract_classes_from_data_set(X_test, y_test, classes_in_data_set[:-1])
pc = PerceptronClassifier(X_vi_se_train, y_vi_se_train, classes_in_data_set[0], classes_in_data_set[1])
score = pc.score(X_vi_se_test, y_vi_se_test)
print('Score {} vs {}: {}'.format(classes_in_data_set[0], classes_in_data_set[1], score))


X_ve_se_train, y_ve_se_train = extract_classes_from_data_set(X_train, y_train, classes_in_data_set[1:])
X_ve_se_test, y_ve_se_test = extract_classes_from_data_set(X_test, y_test, classes_in_data_set[1:])
pc = PerceptronClassifier(X_ve_se_train, y_ve_se_train, classes_in_data_set[1], classes_in_data_set[2])
score2 = pc.score(X_ve_se_test, y_ve_se_test)
print('Score {} vs {}: {}'.format(classes_in_data_set[1], classes_in_data_set[2], score2))

X_vi_ve_train, y_vi_ve_train = extract_classes_from_data_set(X_train, y_train, [classes_in_data_set[0],
                                                             classes_in_data_set[2]])
X_vi_ve_test, y_vi_ve_test = extract_classes_from_data_set(X_test, y_test, [classes_in_data_set[0],
                                                                            classes_in_data_set[2]])
pc = PerceptronClassifier(X_vi_ve_train, y_vi_ve_train, classes_in_data_set[0], classes_in_data_set[2])
score3 = pc.score(X_vi_ve_test, y_vi_ve_test)
print('Score {} vs {}: {}'.format(classes_in_data_set[0], classes_in_data_set[2], score3))
