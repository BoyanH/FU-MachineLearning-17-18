from Classifier import Classifier
from Parser import get_data_set
import numpy as np
import math
from random import random


class FisherClassifier(Classifier):
    def __init__(self, X_train, y_train):
        self.alpha = None
        self.fit(X_train, y_train)

    @staticmethod
    def split_in_classes(X_train, y_train):
        split_X = ([], [])

        for idx, x in enumerate(X_train):
            current_label = y_train[idx]
            split_X[current_label].append(x)

        return split_X

    def project_point(self, x):
        return x.dot(self.alpha)

    def fit(self, X_train, y_train):
        X_a, X_b = FisherClassifier.split_in_classes(X_train, y_train)
        cov_mat_a = np.cov(X_a, rowvar=False, bias=True)
        cov_mat_b = np.cov(X_b, rowvar=False, bias=True)
        center_a = np.array(X_a, dtype=np.float64).mean(0)
        center_b = np.array(X_b, dtype=np.float64).mean(0)

        alpha = np.linalg.pinv(cov_mat_a + cov_mat_b).dot(center_a - center_b)
        alpha_normalized = alpha / np.linalg.norm(alpha)
        self.alpha = alpha_normalized

        projected_center_a = self.project_point(center_a)
        projected_center_b = self.project_point(center_b)
        self.separation_point = (projected_center_a + projected_center_b) / 2

    def predict(self, X):
        return list(map(lambda x: self.predict_single(x), X))

    def predict_single(self, x):
        # project x into alpha (AKA Fisher's vector)
        projected = self.project_point(x)
        return projected < self.separation_point


max_score = 0
min_score = 100
best_seed = 0
worst_seed = 0

for i in range(1000):
    X_train, X_test, y_train, y_test = get_data_set(i)
    classifier = FisherClassifier(X_train, y_train)
    score = classifier.score(X_test, y_test)
    if score > max_score:
        max_score = score
        best_seed = i
    if score < min_score:
        min_score = score
        worst_seed = i

print('Best score for seed={}: {}'.format(best_seed, max_score))
print('Worst score for seed{}: {}'.format(worst_seed, min_score))

# X_train, X_test, y_train, y_test = get_data_set(seed=879)
# classifier = FisherClassifier(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print('Score: {}'.format(classifier.score(X_test, y_test)))
