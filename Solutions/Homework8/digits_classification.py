import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PCA import PCA
from Parser import get_data_set

X_train, X_test, y_train, y_test = get_data_set('digits.data')
pca = PCA(230)
train_transformed = pca.fit_transform(X_train)
test_transformed = pca.transform(X_test)

PCA.plot_variance_for_k(X_train)

prediction_non_transformed = LDA().fit(X_train, y_train).predict(X_test)
score_non_transformed = np.mean(prediction_non_transformed == y_test)
print('Score LDA without PCA: {}'.format(score_non_transformed))

prediction_transformed = LDA().fit(train_transformed, y_train).predict(test_transformed)
score_transformed = np.mean(prediction_transformed == y_test)
print('Score LDA with PCA: {}'.format(score_transformed))

# we merged the data sets to play with different data set splits more easily
# if the data set is split with seed=1, then LDA with PCA yields a score of 100%...
# currently, we use a 70%/30% train/test split
# we found out that 230 components (features) best represent the data set for linear classification
