from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as PCA_sklearn
from PCA import PCA
from Parser import parse_data, get_points_and_labels_from_data

data = parse_data('digits_test.data')
X, _ = get_points_and_labels_from_data(data)
transformed_data = PCA(2).fit_transform(X)

plt.scatter(transformed_data[:,0], transformed_data[:,1])
plt.show()

iris_data = parse_data('iris.data', ',')
X_iris, _ = get_points_and_labels_from_data(iris_data, -1)
transformed_iris_data = PCA(2).fit_transform(X_iris)

plt.scatter(transformed_iris_data[:,1], transformed_iris_data[:,0])
plt.show()

X_SK = PCA_sklearn(n_components=2).fit_transform(X_iris)
plt.scatter(X_SK[:,0], X_SK[:,1])
plt.show()
