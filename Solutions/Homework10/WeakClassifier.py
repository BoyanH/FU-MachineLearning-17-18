from sklearn.linear_model import LinearRegression
import numpy as np
from Classifier import Classifier

class WeakClassifier(Classifier):
    def __init__(self, feature):
        self.feature = feature
        self.lr = LinearRegression()

    def fit(self, X, y):
        X_ = self.transform(X)
        y_ = np.vectorize(lambda x: 1 if x == 1 else -1)(y)
        self.lr.fit(X_, y_)

    def fit_(self, X, y):
        self.lr.fit(X, y)

    def fit_predict(self, X, y):
        X_ = self.transform(X)
        y_ = np.vectorize(lambda x: 1 if x == 1 else -1)(y)
        self.fit_(X_, y_)
        return self.predict_(X_)

    def predict(self, X):
        X_ = self.transform(X)
        return self.predict_(X_)

    def predict_(self, X):
        predictions = self.lr.predict(X)
        return np.vectorize(lambda x: 1 if x > 0 else 0)(predictions)

    def transform(self, X):
        return np.vectorize(lambda x: self.feature.get_value(x),
                            signature='(m,n)->()')(X).reshape(-1,1)

