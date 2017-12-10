from PCA import PCA
from Parser import parse_data, get_points_and_labels_from_data, extract_classes_from_data_set
from matplotlib import pyplot as plt
import numpy as np
import math
import glob
import os
import cv2


def read_pgm(file_name):
    return cv2.imread(file_name, -1)

data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './Dataset'))
files = glob.glob(data_folder + '/lfwcrop_grey/faces/*.pgm')
data_set = []

for file in files:
    image = read_pgm(file)
    data_set.append(image.flatten())

data_set = np.array(data_set)
print('data set parsed')

# drops quickly until about k=320
# but really quickly until about 30, plus we can't really submit 320 eigenfaces for this homework...
# PCA.plot_variance_for_k(data_set, save_plot_name='eigenfaces_variance_for_k')
PCA.plot_variance_for_k(data_set)
print('plotted')


pca_eigenfaces = PCA(30)
# data = parse_data('digits_test.data')
# X, y = get_points_and_labels_from_data(data, 0)
# data_set, _ = extract_classes_from_data_set(X, y, [0])
pca_eigenfaces.fit(data_set)
data_set_dimensions = int(math.sqrt(len(data_set[0])))
principal_components = pca_eigenfaces.transformation_matrix.T

for component in principal_components:
    component += abs(component.min())
    component *= (1.0 / component.max())
    image = np.reshape(component, (data_set_dimensions, data_set_dimensions)).astype(np.float32).T

    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()
