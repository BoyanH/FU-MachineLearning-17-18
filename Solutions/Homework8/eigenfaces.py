from PCA import PCA
import numpy as np
import glob
import os


def read_pgm(file):
    infile = open(file, 'r')
    image = np.fromfile(infile, dtype=np.uint16)

    return image


data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './Dataset'))
files = glob.glob(data_folder + '/lfwcrop_grey/faces/*.pgm')
data_set = []

for file in files:
    data_set.append(read_pgm(file))

data_set = np.array(data_set)

# drops quickly until about k=320
# therefore, for better visualization, as math.sqrt(320) = 17.88
# we decided to use k = 18^2 to be able to draw good square images of the principal components
PCA.plot_variance_for_k(data_set, save_plot_name='eigenfaces_variance_for_k')
