from Classifier import Classifier
from Parser import get_data_set
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from random import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class FisherClassifier(Classifier):
    def __init__(self, X_train, y_train):
        self.alpha = None
        self.threshold = None
        self.fit(X_train, y_train)

    @staticmethod
    def split_in_classes(X_train, y_train):
        split_X = ([], [])

        for idx, x in enumerate(X_train):
            current_label = y_train[idx]
            split_X[current_label].append(x)

        return split_X

    @staticmethod
    def get_density_function(center, covariance):
        return lambda x: math.e ** (
            (-1/2) * ((x - center) / covariance) ** 2
        ) / (covariance * math.sqrt((2*math.pi)))

    @staticmethod
    def get_covariance_for_projected(points, center):
        vectorized_sq_distances_sum = np.vectorize(lambda x, m: (x - m)**2)
        square_distances_sum = np.sum(vectorized_sq_distances_sum(points, center))
        return math.sqrt(square_distances_sum / len(points))

    def plot_class(self, points, center):
        projected_center = self.project_point(center)
        # project_points = np.vectorize(lambda x:  self.project_point(x), signature='(n)->(n)')
        projected_points = list(map(lambda x: self.project_point(x), points))
        covariance = FisherClassifier.get_covariance_for_projected(projected_points, projected_center)
        density_a = FisherClassifier.get_density_function(projected_center, covariance)
        plot_distance = 5000
        y_of_plot = [density_a(float(x) / 100) for x in range(-plot_distance, plot_distance)]
        x_to_plot = [float(x) / 100 for x in range(-plot_distance, plot_distance)]
        first_decent = None
        last_decent = None

        for idx, y in enumerate(y_of_plot):
            if y > 0.01 and first_decent is None:
                first_decent = idx
            elif y <= 0.01 and first_decent is not None:
                last_decent = idx
                break

        beauty_margin = (last_decent - first_decent)
        start = int(first_decent - beauty_margin/2)
        end = int(last_decent + beauty_margin/2)

        plt.plot(x_to_plot[start:end], y_of_plot[start:end])

    def plot_probability_distribution(self, center_a, center_b, points_a, points_b):
        self.plot_class(points_a, center_a)
        self.plot_class(points_b, center_b)
        plt.show()

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

        # to determine whether a point belongs to class a or class b we need a threshold
        # on the 1 dimensional space. This one is the projected point between the 2 centers
        self.threshold = self.project_point((center_a + center_b) / 2)

        self.plot_probability_distribution(center_a, center_b, X_a, X_b)

    def predict(self, X):
        return list(map(lambda x: self.predict_single(x), X))

    def predict_single(self, x):
        # project x into alpha (AKA Fisher's vector)
        projected = self.project_point(x)
        return projected < self.threshold


# max_score = 0
# min_score = 100
# best_seed = 0
# worst_seed = 0

# for i in range(1000):
#     X_train, X_test, y_train, y_test = get_data_set(i)
#     classifier = FisherClassifier(X_train, y_train)
#     score = classifier.score(X_test, y_test)
#     if score > max_score:
#         max_score = score
#         best_seed = i
#     if score < min_score:
#         min_score = score
#         worst_seed = i
#
# print('Best score for seed={}: {}'.format(best_seed, max_score))
# print('Worst score for seed{}: {}'.format(worst_seed, min_score))

X_train, X_test, y_train, y_test = get_data_set(879)
classifier = FisherClassifier(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Score: {}'.format(classifier.score(X_test, y_test)))

lm = LinearRegression()
y_train_modified = list(map(lambda x: 1 if x == 1 else -1, y_train))
lm.fit(X_train, y_train_modified)
prediction = np.array(list(map(lambda x: 1 if x > 0 else 0, lm.predict(X_test))), dtype=np.float64)
score = np.mean(prediction == np.array(y_test, dtype=np.float64))
print('Score with linear regression: {}'.format(score))
