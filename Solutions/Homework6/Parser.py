import csv
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def parse_data():
    file_name = os.path.join(os.path.dirname(__file__), './Dataset/iris.data')
    return pd.read_csv(file_name, header=None).as_matrix()


def get_points_and_labels_from_data(data):
    points = data[:,:-1].astype('float')
    labels = data[:,-1]

    return points, labels


def extract_classes_from_data_set(X, y, classes):
    is_from_classes = np.vectorize(lambda y: y in classes)
    filter_arr = is_from_classes(y)
    return X[filter_arr], y[filter_arr]


def get_data_set(seed):
    data = parse_data()
    X, y = get_points_and_labels_from_data(data)
    # for determined results we use a seed for random_state, so that data is always split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                        random_state=seed)

    return X_train, X_test, y_train, y_test
