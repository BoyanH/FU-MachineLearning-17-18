import numpy as np
from Classifier import Classifier
from Parser import get_test_set, get_train_set
from Helpers import get_integral_image
from FDFeature import FDFeature
from FDFType import FDFType
from WeakClassifier import WeakClassifier
from AdaBoost import AdaBoost

types = [FDFType.TWO_RECTANGLE_HORIZONTAL, FDFType.TWO_RECTANGLE_VERTICAL,
         FDFType.THREE_RECTANGLE_HORIZONTAL, FDFType.THREE_RECTANGLE_VERTICAL,
         FDFType.FOUR_RECTANGLE]


class FaceDetection(Classifier):
    def __init__(self, k=300):
        self.classifiers = []
        self.cw = []  # classifier weights
        self.k_classifiers = k

    def fit(self, X, y):
        X_ = self.transform(X)

        # initialize all possible classifiers
        possible_classifiers = []
        classifier_predictions = []
        percents = np.arange(.4, .65, .05)

        # for a number of various feature positions, sizes and types,
        # create the pool of classifiers which will later be sieved with AdaBoost
        # Total amount of classifiers (features for face detection) is 405
        for wp in percents:
            for hp in percents:
                for xp in np.arange(0.1, 1 - wp, .1):
                    for yp in np.arange(0.1, 1 - hp, .1):
                        for f_type in types:
                            feature = FDFeature(wp, hp, f_type, xp, yp)
                            classifier = WeakClassifier(feature)
                            classifier_predictions.append(classifier.fit_predict(X_, y))
                            possible_classifiers.append(classifier)

        print('{} weak classifiers were successfully trained and their predictions saved!'.format(
            len(possible_classifiers)))

        self.classifiers, self.cw = AdaBoost.boost_classifiers(
            possible_classifiers, classifier_predictions, y, self.k_classifiers)

    def transform(self, X):
        return np.vectorize(lambda x: get_integral_image(x),
                            signature='(m,n)->(z,c)')(X)

    def predict(self, X):
        X_ = self.transform(X)
        predictions = np.zeros(len(X))

        for i, classifier in enumerate(self.classifiers):
            predictions += classifier.predict(X_) * self.cw[i]

        weights_sum = self.cw.sum()
        return np.vectorize(lambda p: 1 if p >= weights_sum else 0)(predictions)


X_test, y_test = get_test_set()
X_train, y_train = get_train_set()

ds_size = None
fd = FaceDetection()
fd.fit(X_train[:ds_size], y_train[:ds_size])
predictions_train = fd.predict(X_train[:ds_size])
predictions_test = fd.predict(X_test[:ds_size])
print('Score train: {}'.format(fd.score(predictions_train, y_train[:ds_size])))
print('Score test: {}'.format(fd.score(predictions_test, y_test[:ds_size])))
print('Confusion matrix train: \n{}'.format(fd.confusion_matrix(predictions_train, y_train[:ds_size])))
print('Confusion matrix test: \n{}'.format(fd.confusion_matrix(predictions_test, y_test[:ds_size])))

# print(X_test[0].shape)
#
# feature = FDFeature(.7, .3, FDFType.TWO_RECTANGLE_HORIZONTAL, .2, .2)
# wc = WeakClassifier(feature)
# X_train_ = np.vectorize(lambda x: get_integral_image(x),
#                         signature='(m,n)->(z,c)')(X_train)
# wc.fit(X_train_, y_train)
# print(wc.score(X_train_, y_train))
