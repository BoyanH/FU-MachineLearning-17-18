import numpy as np

class Classifier:
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)