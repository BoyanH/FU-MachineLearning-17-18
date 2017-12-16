import numpy as np
import math
import os
from matplotlib import pyplot as plt


def plot_multiple_images(title, images, data_set_dimensions, cols=5, fig_name=None):
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

    if fig_name is not None:
        file_name = os.path.join(os.path.dirname(__file__), './{}'.format(fig_name))
        plt.savefig(file_name)
    else:
        plt.show()