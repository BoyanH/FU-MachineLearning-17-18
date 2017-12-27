import numpy as np
import glob
import os
import cv2

def read_pgm(file_name):
    return cv2.imread(file_name, -1)


def get_data_set(data_set_type='test'):
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './Dataset'))
    faces = glob.glob(data_folder + '/{}/face/*.pgm'.format(data_set_type))
    non_faces = glob.glob(data_folder + '/{}/non-face/*.pgm'.format(data_set_type))
    data_set = []
    labels = []

    for file in faces:
        image = read_pgm(file)
        data_set.append(image)

    for file in non_faces:
        image = read_pgm(file)
        data_set.append(image)

    labels = labels + ([1]*len(faces))
    labels = labels + ([0]*len(non_faces))
    labels = np.array(labels)

    data_set = np.array(data_set)
    np.random.seed(1)
    idx = np.random.permutation(len(data_set))

    # return a random permutation so positive and negative samples are mixed
    return data_set[idx], labels[idx]


def get_test_set():
    return get_data_set('test')


def get_train_set():
    return get_data_set('train')
