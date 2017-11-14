from Classifier import Classifier
from Parser import get_data_set
import numpy as np
import math


class FisherClassifier(Classifier):

    def __init__(self, X_train, y_train):
        self.fit(X_train, y_train)

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        return list(map(lambda x: self.predict_single(x), X))

    def predict_single(self, point):
        pass

X_train, X_test, y_train, y_test = get_data_set()
