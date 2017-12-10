from PCA import PCA
from matplotlib import pyplot as plt
import numpy as np
import math
import glob
import os
import cv2


image_shape = (16,16)


def plot_eigenfaces(title, images, cols=5):
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=(8,8))
    plt.suptitle(title, size=20)

    for i, component in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        component += abs(component.min())
        component *= (1.0 / component.max())
        image = np.reshape(component, (data_set_dimensions, data_set_dimensions))
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xticks(())
        plt.yticks(())

    plt.show()

def read_pgm(file_name):
    return cv2.imread(file_name, -1)

data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './Dataset'))
files = glob.glob(data_folder + '/lfwcrop_grey/faces/*.pgm')
data_set = []

for file in files:
    image = read_pgm(file)
    data_set.append(image.flatten())

data_set = np.array(data_set)

# drops quickly until about k=320
# but really quickly until about 30, plus we can't really submit 320 eigenfaces for this homework...
# PCA.plot_variance_for_k(data_set, save_plot_name='eigenfaces_variance_for_k')


pca_eigenfaces = PCA(30)
pca_eigenfaces.fit(data_set)
data_set_dimensions = int(math.sqrt(len(data_set[0])))
principal_components = pca_eigenfaces.principal_components

plot_eigenfaces('Eigenfaces', principal_components)
