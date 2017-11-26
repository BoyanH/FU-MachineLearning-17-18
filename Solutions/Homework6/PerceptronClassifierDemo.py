import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PerceptronClassifier import PerceptronClassifier, classes_in_data_set
from Parser import get_data_set, extract_classes_from_data_set

# Classes
# Iris-setosa, Iris-versicolour, Iris-virginica

X_train, X_test, y_train, y_test = get_data_set(1)

X_vi_se_train, y_vi_se_train = extract_classes_from_data_set(X_train, y_train, classes_in_data_set[:-1])
X_vi_se_test, y_vi_se_test = extract_classes_from_data_set(X_test, y_test, classes_in_data_set[:-1])
pc_vi_se = PerceptronClassifier(X_vi_se_train, y_vi_se_train, classes_in_data_set[0], classes_in_data_set[1])
score_vi_se = pc_vi_se.score(X_vi_se_test, y_vi_se_test)
score_vi_se_train = pc_vi_se.score(X_vi_se_train, y_vi_se_train)
print('Score {} vs {}: {}'.format(classes_in_data_set[0], classes_in_data_set[1], score_vi_se))
print('Score {} vs {} on train data: {}'.format(classes_in_data_set[0], classes_in_data_set[1], score_vi_se_train))


X_ve_se_train, y_ve_se_train = extract_classes_from_data_set(X_train, y_train, classes_in_data_set[1:])
X_ve_se_test, y_ve_se_test = extract_classes_from_data_set(X_test, y_test, classes_in_data_set[1:])
pc_ve_se = PerceptronClassifier(X_ve_se_train, y_ve_se_train, classes_in_data_set[1], classes_in_data_set[2])
score_ve_se = pc_ve_se.score(X_ve_se_test, y_ve_se_test)
print('Score {} vs {}: {}'.format(classes_in_data_set[1], classes_in_data_set[2], score_ve_se))

X_vi_ve_train, y_vi_ve_train = extract_classes_from_data_set(X_train, y_train, [classes_in_data_set[0],
                                                             classes_in_data_set[2]])
X_vi_ve_test, y_vi_ve_test = extract_classes_from_data_set(X_test, y_test, [classes_in_data_set[0],
                                                                            classes_in_data_set[2]])
pc_vi_ve = PerceptronClassifier(X_vi_ve_train, y_vi_ve_train, classes_in_data_set[0], classes_in_data_set[2])
score_vi_ve = pc_vi_ve.score(X_vi_ve_test, y_vi_ve_test)
score_vi_ve_train = pc_vi_ve.score(X_vi_ve_train, y_vi_ve_train)
print('Score {} vs {}: {}'.format(classes_in_data_set[0], classes_in_data_set[2], score_vi_ve))
print('Score {} vs {} on train data: {}'.format(classes_in_data_set[0], classes_in_data_set[2], score_vi_ve_train))

clf = LinearDiscriminantAnalysis()
clf.fit(X_vi_ve_train, y_vi_ve_train)
predictions = clf.predict(X_vi_ve_test)
predictions_train = clf.predict(X_vi_ve_train)
score_lda = np.mean(predictions == y_vi_ve_test)
score_lda_train = np.mean(predictions_train == y_vi_ve_train)
print('Score of LDA {} vs {}: {}'.format(classes_in_data_set[0], classes_in_data_set[1], score_lda))
print('Score of LDA {} vs {} on train data: {}'.format(classes_in_data_set[0], classes_in_data_set[1], score_lda_train))
