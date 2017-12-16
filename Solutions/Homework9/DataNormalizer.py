import numpy as np

class DataNormalizer:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.var

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)