from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from NN import NN

print('fetching data-set')
mnist = fetch_mldata('mnist-original', data_home='./Dataset/mnist.pkl/mnist_dataset/')
print('MNIST original fetched')
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3,
                                                        random_state=1)

nn = NN()
nn.fit(X_train, y_train)
nn.plot_accuracies('./20_inner_big.png')
print('Score train: {}'.format(nn.score(X_train, y_train)))
print('Score: {}'.format(nn.score(X_test, y_test)))
