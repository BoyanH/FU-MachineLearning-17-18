from Classifier import Classifier
from Parser import *
import numpy as np
import math


class GaussianClassifier(Classifier):
    @staticmethod
    def covariance_for_point(point, center):
        # calculate covariance for a single point (well not really, but a single summar thingy)
        return np.matrix(point - center).T.dot(np.matrix(point - center))

    def __init__(self, train_data, classes = [x for  x in range(10)]):
        """
        :param classes: list of classes the classifier should train itself to distinguish
                        (e.g [3,5] for 3 vs 5 classifier)
        :param trainData:
        :param trainLabels:
        :param testData:
        :param testLabels:
        """

        self.centers = {}
        self.covariance_matrix = {}
        self.covariance_matrix_det = {}
        self.convariance_matrix_pinv = {}
        (train_labels, train_data) = get_labels_and_points_from_data(train_data, classes)
        self.classes = classes
        self.fit(train_labels, train_data)

    def fit(self, train_labels, train_data):
        assert(len(train_labels) == len(train_data))
        points_per_label = {}

        for idx, point in enumerate(train_data):
            current_label = train_labels[idx]
            if not current_label in points_per_label:
                points_per_label[current_label] = [point]
            else:
                points_per_label[current_label].append(point)

        for label in points_per_label:
            # average of all points from the current class
            self.centers[label] = np.array(points_per_label[label]).mean(0)
            # calculate covariance matrix using vectorization (see covariance_for_point static method)
            self.covariance_matrix[label] = np.vectorize(GaussianClassifier.covariance_for_point, signature='(m),(n)->(m,m)')(
                points_per_label[label], self.centers[label]).sum(axis=0) / len(points_per_label[label])
            self.covariance_matrix_det[label] = np.linalg.det(self.covariance_matrix[label])
            self.convariance_matrix_pinv[label] = np.linalg.pinv(self.covariance_matrix[label])

    def predict(self, X):
        return list(map(lambda x: self.predict_single(x), X))

    def predict_single(self, point):
        possibilities = list(map(lambda x: self.get_possibility_for_class(x, point), self.classes))
        winningIndex = possibilities.index(max(possibilities))


        return self.classes[winningIndex]

    def get_possibility_for_class(self, point_class, point):
        # using the formula from the lecture, calculate the probability for a point with coordinates to
        # be part of a class

        # only important thing here is that 2*pi*det(covariance_matrix) can be zero, so we use
        # np.nextafter to replace any zeros with a reaaaaly small float (because of DivideByZero exceptions...)

        two_pi_det = 2 * math.pi * self.covariance_matrix_det[point_class]
        left_side = 1 / max(np.nextafter(np.float16(0), np.float16(1)), math.sqrt(two_pi_det))
        right_side = math.e**(-0.5 * (point - self.centers[point_class]).T.
                              dot(self.convariance_matrix_pinv[point_class]).dot(point - self.centers[point_class]))
        return left_side * right_side

train_data = parse_data_file('./Dataset/train')
test_data = parse_data_file('./Dataset/test')

three_vs_five = GaussianClassifier(train_data, [3,5])
(three_vs_five_test_labels, three_vs_five_test_data) = get_labels_and_points_from_data(test_data, [3,5])
print("Score 3 vs 5: {}%".format(three_vs_five.score(three_vs_five_test_data, three_vs_five_test_labels)))

three_vs_seven = GaussianClassifier(train_data, [3,7])
(three_vs_seven_test_labels, three_vs_seven_test_data) = get_labels_and_points_from_data(test_data, [3,7])
print("Score 3 vs 7: {}%".format(three_vs_seven.score(three_vs_seven_test_data, three_vs_seven_test_labels)))

three_vs_eight = GaussianClassifier(train_data, [3,8])
(three_vs_eight_test_labels, three_vs_eight_test_data) = get_labels_and_points_from_data(test_data, [3,8])
print("Score 3 vs 8: {}%".format(three_vs_eight.score(three_vs_eight_test_data, three_vs_eight_test_labels)))

five_vs_seven = GaussianClassifier(train_data, [5,7])
(five_vs_seven_test_labels, five_vs_seven_test_data) = get_labels_and_points_from_data(test_data, [5,7])
print("Score 5 vs 7: {}%".format(five_vs_seven.score(five_vs_seven_test_data, five_vs_seven_test_labels)))

five_vs_eight = GaussianClassifier(train_data, [5,8])
(five_vs_eight_test_labels, five_vs_eight_test_data) = get_labels_and_points_from_data(test_data, [5,8])
print("Score 5 vs 8: {}%".format(five_vs_eight.score(five_vs_eight_test_data, five_vs_eight_test_labels)))

seven_vs_eight = GaussianClassifier(train_data, [7,8])
(seven_vs_eight_test_labels, seven_vs_eight_test_data) = get_labels_and_points_from_data(test_data, [7,8])
print("Score 7 vs 8: {}%".format(seven_vs_eight.score(seven_vs_eight_test_data, seven_vs_eight_test_labels)))

combined = GaussianClassifier(train_data, [3, 5, 7, 8])
(combined_test_labels, combined_test_data) = get_labels_and_points_from_data(test_data, [3,5,7,8])
print("Score 3 vs 5 vs 7 vs 8: {}%".format(combined.score(combined_test_data, combined_test_labels)))

all_digits = [x for x in range(10)]
all = GaussianClassifier(train_data, all_digits)
(all_test_labels, all_test_data) = get_labels_and_points_from_data(test_data, all_digits)
print("Score all: {}%".format(combined.score(all_test_data, all_test_labels)))
