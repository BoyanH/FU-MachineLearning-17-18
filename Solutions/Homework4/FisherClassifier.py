from Classifier import Classifier
from Parser import get_data_set
import numpy as np
import math


class FisherClassifier(Classifier):
    def __init__(self, X_train, y_train):
        self.alpha = None
        self.fit(X_train, y_train)

    @staticmethod
    def split_in_classes(X_train, y_train):
        splited_X = ([], [])

        for idx, x in enumerate(X_train):
            current_label = y_train[idx]
            splited_X[current_label].append(x)

        return splited_X

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


X_train, X_test, y_train, y_test = get_data_set()
classifier = FisherClassifier(X_train, y_train)
print('Score: {}'.format(classifier.score(X_test, y_test)))
