from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as PCA_sklearn
from PCA import PCA
from Parser import parse_data, get_points_and_labels_from_data, extract_classes_from_data_set

data = parse_data('digits_test.data')
X, y = get_points_and_labels_from_data(data)

for a in range(10):
    for b in range(a+1, 10, 1):
        X_extracted, y_extracted = extract_classes_from_data_set(X, y, [a,b])
        X_transformed = PCA(2).fit_transform(X_extracted)
        X_a_idx = y_extracted.astype(int) == a
        X_a = X_transformed[X_a_idx]
        X_b = X_transformed[np.invert(X_a_idx)]

        plt.scatter(X_a[:,0], X_a[:,1], c='#0000FF')
        plt.scatter(X_b[:,0], X_b[:,1], c='#00FF00')
        plt.xlabel('{} vs {}'.format(a, b), fontsize=24)
        plt.savefig('figs/{}vs{}.png'.format(a,b))
        plt.clf()
