import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split


def parse_data(name):
    file_name = os.path.join(os.path.dirname(__file__), './Dataset/{}.data'.format(name))
    csv_file = open(file_name, 'rt')
    reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
    data = []

    for row in reader:
        filtered = list(filter(lambda x: x != '', row))
        data.append(list(map(lambda x: float(x), filtered)))

    return data


def get_points_and_labels_from_data(data):
    points = np.array(list(map(lambda x: x[:-1], data)), dtype=np.float64)
    labels = np.array(list(map(lambda x: int(x[-1]), data)))

    return points, labels

def extract_classes_from_data_set(X, y, classes):
    is_from_classes = np.vectorize(lambda y: y in classes)
    filter_arr = is_from_classes(y)
    return X[filter_arr], y[filter_arr]


def get_data_set(seed):
    data = parse_data('spambase')
    X, y = get_points_and_labels_from_data(data)
    # for determined results we use a seed for random_state, so that data is always split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                        random_state=seed)

    return X_train, X_test, y_train, y_test
