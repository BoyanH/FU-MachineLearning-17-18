import csv
import numpy as np


def parse_data_file(file_name):
    file = open(file_name, 'rt')
    reader = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONE)
    data = []

    for row in reader:
        filtered = list(filter(lambda x: x != '', row))
        data.append(list(map(lambda x: float(x), filtered)))

    return data


def get_labels_and_points_from_data(data, classes):
    data = list(filter(lambda x: int(x[0]) in classes, data))
    labels = np.array(list(map(lambda x: int(x[0]), data)))
    points = np.array(list(map(lambda x: x[1:], data)))

    return labels, points
