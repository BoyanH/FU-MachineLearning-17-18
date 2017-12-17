from Parser import get_data_set
from NN import NN


X_train, X_test, y_train, y_test = get_data_set('digits.data')
nn = NN()
nn.fit(X_train, y_train)
nn.plot_accuracies()
print('Score train: {}'.format(nn.score(X_train, y_train)))
print('Score: {}'.format(nn.score(X_test, y_test)))
