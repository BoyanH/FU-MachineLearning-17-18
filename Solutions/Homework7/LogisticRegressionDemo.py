from sklearn.linear_model import LogisticRegression as LRSKL
import numpy as np
from Parser import get_data_set
from LogisticRegression import LogisticRegression

X_train, X_test, y_train, y_test = get_data_set(1)
lr = LogisticRegression(X_train, y_train, plot=True)
score = lr.score(X_test, y_test)
print('Score: {}'.format(score))
print(lr.confusion_matrix(X_test, y_test))

sklearn_lr = LRSKL()
sklearn_lr.fit(X_train, y_train)
predictions = sklearn_lr.predict(X_test)
score_sklearn = np.mean(predictions == y_test)
print('Score from sklearn: {}'.format(score_sklearn))
