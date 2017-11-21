import os
import pandas as pd
from numpy import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt


def save_plot(fig, path):
    fig.savefig(os.path.join(os.path.dirname(__file__), path))


def plot_covariance(ax1, x_initial, y_initial, cov):
    num_points = 1000
    radius = 1.5

    arcs = np.linspace(0, 2 * pi, num_points)
    x = radius * sin(arcs) + x_initial
    y = radius * cos(arcs) + y_initial

    # TODO: save covariance and inverted covariance matrix separately
    xy = np.array(list(zip(x, y)))
    x, y = zip(*xy.dot(np.linalg.pinv(cov)))

    ax1.plot(x, y)
