import csv
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def parse_data(file_name, sep=' '):
    file_name = os.path.join(os.path.dirname(__file__), './Dataset/{}'.format(file_name))
    return pd.read_csv(file_name, header=None, na_filter=True, sep=sep).as_matrix()


def get_points_and_labels_from_data(data, label_idx=0):
    points = (np.array(data[:,label_idx:], dtype=np.float64) if label_idx == 0 else
              np.array(data[:, :label_idx], dtype=np.float64))
    labels = np.array(data[:,label_idx])

    return points, labels


def get_data_set(file_name, sep=' ', label_idx=0, seed=1):
    data = parse_data(file_name, sep)
    X, y = get_points_and_labels_from_data(data, label_idx)
    # for determined results we use a seed for random_state, so that data is always split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                        random_state=seed)

    return X_train, X_test, y_train, y_test
