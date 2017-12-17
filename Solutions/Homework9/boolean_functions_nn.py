import numpy as np
from NN import NN


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y_and = np.array([a & b for a,b in X])
y_or  = np.array([a | b for a,b in X])
y_xor = np.array([a ^ b for a,b in X])

# AND
nn = NN(max_iterations=3000, size_hidden=10, size_output=2, learning_rate=0.01)
nn.fit(X, y_and)
nn.plot_accuracies('./nn_and.png')
print('Score and: {}'.format(nn.score(X, y_and)))

# OR
nn = NN(max_iterations=3000, size_hidden=10, size_output=2, learning_rate=0.01)
nn.fit(X, y_or)
nn.plot_accuracies('./nn_or.png')
print('Score or: {}'.format(nn.score(X, y_or)))

# XOR
nn = NN(max_iterations=3000, size_hidden=10, size_output=2, learning_rate=0.01)
nn.fit(X, y_and)
nn.plot_accuracies('./nn_xor.png')
print('Score xor: {}'.format(nn.score(X, y_and)))
