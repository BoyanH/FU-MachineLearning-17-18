from operator import itemgetter
from Classifier import Classifier
from Parser import *
import numpy as np
import math


class GaussianClassifier(Classifier):
    @staticmethod
    def covariance_for_point(point, center):
        # in lecture, formula was (Xi - M)(Xi - M)^T
        # but what we mean by Xi - M is a vector and what numpy means is a
        # 1xn matrix (1 row, n columns), therefore we need to transpose the left side
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

        (train_labels, train_data) = get_labels_and_points_from_data(train_data, classes)
        self.classes = classes
        self.fit(train_labels, train_data)

    def fit(self, train_labels, train_data):
        assert(len(train_labels) == len(train_data))
        points_per_label = {}
        self.centers = {}
        self.covariance_matrix = {}

        for idx, point in enumerate(train_data):
            current_label = train_labels[idx]
            if not current_label in points_per_label:
                points_per_label[current_label] = [point]
            else:
                points_per_label[current_label].append(point)

        for label in points_per_label:
            # average of all points from the current class
            self.centers[label] = np.array(points_per_label[label]).mean(0)
            # TODO: comment
            self.covariance_matrix[label] = np.vectorize(GaussianClassifier.covariance_for_point, signature='(m),(n)->(m,m)')(
                points_per_label[label], self.centers[label]).sum(axis=0) / len(points_per_label[label])

    def predict(self, X):
        return list(map(lambda x: self.predict_single(x), X))

    def predict_single(self, point):
        possibilities = list(map(lambda x: self.get_possibility_for_class(x, point), self.classes))
        winningIndex = possibilities.index(max(possibilities))


        return self.classes[winningIndex]

    def get_possibility_for_class(self, point_class, point):
        two_pi_det = 2 * math.pi * np.linalg.det(self.covariance_matrix[point_class])
        left_side = 1 / max(np.nextafter(np.float16(0), np.float16(1)), math.sqrt(two_pi_det))
        right_side = math.e**(-0.5 * (point - self.centers[point_class]).T.
                              dot(np.linalg.pinv(self.covariance_matrix[point_class])).dot(point - self.centers[point_class]))
        return left_side * right_side

train_data = parse_data_file('./Dataset/train')
test_data = parse_data_file('./Dataset/test')

three_vs_five = GaussianClassifier(train_data, [3,5])
(test_labels, test_data) = get_labels_and_points_from_data(test_data, [3,5])
print(three_vs_five.score(test_data, test_labels))