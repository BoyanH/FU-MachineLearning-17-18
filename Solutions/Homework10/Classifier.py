import numpy as np

class Classifier:
    def score(self, predictions, y):
        return np.mean(predictions == y)
    
    def confusion_matrix(self, predicted, y):
        size = 2 # hardcoded here
        results = np.zeros((size, size), dtype=np.int32)

        for pi, yi in zip(predicted, y):
            results[int(pi)][int(yi)] += 1

        return results
