import os
import pandas as pd
from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt

RGB_BLACK = [0, 0, 0]


def save_plot(fig, path):
    fig.savefig(os.path.join(os.path.dirname(__file__), path))


def plot_covariance(ax1, x_initial, y_initial, cov):
    num_points = 1000
    radius = 1.5  # adjusted radius, seems more correct this way

    # plot a circle
    arcs = np.linspace(0, 2 * pi, num_points)
    x = radius * sin(arcs)
    y = radius * cos(arcs)

    # stretch it according to the covariance matrix
    xy = np.array(list(zip(x, y)))
    x, y = zip(*xy.dot(cov))

    # move it in the space so it's center is above the cluster's center
    x = x + x_initial
    y = y + y_initial

    ax1.scatter(x, y, c=RGB_BLACK, s=10)  # plot covariance
    ax1.scatter([x_initial], [y_initial], c=RGB_BLACK, s=50)  # plot center
