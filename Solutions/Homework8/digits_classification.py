import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PCA import PCA
from Parser import get_data_set

X_train, X_test, y_train, y_test = get_data_set('digits.data')

pca_230 = PCA(230)
train_transformed_230 = pca_230.fit_transform(X_train)
test_transformed_230 = pca_230.transform(X_test)

pca_30 = PCA(30)
train_transformed_30 = pca_30.fit_transform(X_train)
test_transformed_30 = pca_30.transform(X_test)

PCA.plot_variance_for_k(X_train)

prediction_non_transformed = LDA().fit(X_train, y_train).predict(X_test)
score_non_transformed = np.mean(prediction_non_transformed == y_test)
print('Score LDA without PCA: {}'.format(score_non_transformed))

prediction_transformed_230 = LDA().fit(train_transformed_230, y_train).predict(test_transformed_230)
score_transformed_230 = np.mean(prediction_transformed_230 == y_test)
print('Score LDA with PCA, k={}: {}'.format(230, score_transformed_230))

prediction_transformed_30 = LDA().fit(train_transformed_30, y_train).predict(test_transformed_30)
score_transformed_30 = np.mean(prediction_transformed_30 == y_test)
print('Score LDA with PCA, k={}: {}'.format(30, score_transformed_30))
