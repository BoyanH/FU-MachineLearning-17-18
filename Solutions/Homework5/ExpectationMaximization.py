import numpy as np
from random import random
import random
import math

CHOOSE_INITIAL_CENTERS_RANDOMLY = True
USE_DISTANCE_THRESHOLD = True
MOVEMENT_THRESHOLD = 0.002
DISTANCE_THRESHOLD = 0.01

VERBOSE = False


class ExpectationMaximization:
    @staticmethod
    def get_initial_centers_from_data_set(data, k):
        if CHOOSE_INITIAL_CENTERS_RANDOMLY:
            random.seed(8)
            return np.array(random.choices(data, k=k), dtype=np.float64)

        min_point = data.min(0)
        max_point = data.max(0)
        centers = []

        for i in range(k):
            centers.append(min_point + (max_point - min_point) / k)

        return centers

    def __init__(self):
        self.data = None
        self.k_clusters = None
        self.sigma = None
        self.cluster_centers = None
        self.points_per_cluster = None
        self.inv_covariances_per_cluster = []
        self.covariances_per_cluster = []
        self.cluster_indexes = None
        self.last_diff = None
        self.mean_distance = None
        self.max_iterations = None

    def reset_points_per_cluster(self):
        self.points_per_cluster = [[x] for x in self.cluster_centers]

    def cluster(self, data, k_clusters, max_iterations=30):
        self.data = data
        self.mean_distance = None
        self.last_diff = None
        self.k_clusters = k_clusters
        self.cluster_indexes = [x for x in range(self.k_clusters)]
        self.cluster_centers = ExpectationMaximization.get_initial_centers_from_data_set(data, k_clusters)
        self.covariances_per_cluster = [[np.identity(len(x))] for x in self.cluster_centers]
        self.inv_covariances_per_cluster = self.covariances_per_cluster
        self.reset_points_per_cluster()
        self.max_iterations = max_iterations
        self.apply_expectation_maximization()

    def apply_expectation_maximization(self, k=0):
        if VERBOSE:
            print('iteration: {}'.format(k))
        old_centers = np.copy(self.cluster_centers)
        old_distance = np.copy(self.mean_distance)
        self.reset_points_per_cluster()
        self.assign_points_to_clusters()
        self.calculate_cluster_centers()
        self.calculate_covariances()
        self.update_mean_distance_to_cluster_centers()

        # for some reason old_distance is None or old_distance > ...
        # was throwing errors cannot compare NoneType with int
        # therefore the less readable not old_distance :X

        # but basically, if judging on distance for when to stop,
        # if the average distance gets worse, we reach the max amount of iterations or we reach our desired
        # threshold, stop
        if USE_DISTANCE_THRESHOLD:
            if (self.mean_distance > DISTANCE_THRESHOLD and
                        k < self.max_iterations and (not old_distance or old_distance > self.mean_distance)):
                self.apply_expectation_maximization(k + 1)

        # other method to determine when to stop is by simply checking if the cluster centers still move enough
        elif abs((old_centers - self.cluster_centers).sum()) > MOVEMENT_THRESHOLD and k < self.max_iterations:
            self.apply_expectation_maximization(k + 1)

    def update_mean_distance_to_cluster_centers(self):
        # 1. get distance for every point in each cluster in a single array
        # 2. get mean of that

        dis_for_x_in_k = np.vectorize(lambda x, i: self.get_distance_mahalanobis_to_cluster(x, i),
                                      signature='(m),()->()')

        # didn't manage to vectorize this one ;/ somehow, numpy doens't like jagged arrays
        distances_for_points = list(map(lambda x: dis_for_x_in_k(self.points_per_cluster[x], x),
                                        self.cluster_indexes))

        flattened_distances = []
        for cluster_distances in distances_for_points:
            flattened_distances = flattened_distances + list(cluster_distances)

        self.mean_distance = np.array(flattened_distances, dtype=np.float64).mean()

        if VERBOSE:
            print('Average Mahalanobis\' distance to cluster center: {}'.format(self.mean_distance))

    def assign_points_to_clusters(self):
        assign_points_to_clusters = np.vectorize(lambda x: self.assign_point_to_cluster(x),
                                                 signature='(m)->()')
        assign_points_to_clusters(self.data)

        for idx, list in enumerate(self.points_per_cluster):
            self.points_per_cluster[idx] = np.array(self.points_per_cluster[idx])

    def assign_point_to_cluster(self, point):
        distances_to_clusters = [self.get_distance_mahalanobis_to_cluster(point, i) for i in self.cluster_indexes]
        closest_cluster_idx = np.argmin(distances_to_clusters)
        self.points_per_cluster[closest_cluster_idx].append(point)

    def get_distance_mahalanobis_to_cluster(self, x, i_cluster):
        # the square root could be removed, but helps for better plotting
        return math.sqrt((x - self.cluster_centers[i_cluster]).dot(
            self.inv_covariances_per_cluster[i_cluster]).dot((x - self.cluster_centers[i_cluster]).T))

    def calculate_cluster_centers(self):
        get_cluster_centers = np.vectorize(lambda x, points_per_center: points_per_center[x].mean(0),
                                           signature='(),(m)->(n)')
        self.cluster_centers = get_cluster_centers(self.cluster_indexes, self.points_per_cluster)

    def calculate_covariances(self):
        # if we only have the center in our cluster (empty cluster) then just use the identity
        # matrix as covariance matrix again
        get_covariances = np.vectorize(lambda x, points_for_cluster: np.cov(points_for_cluster[x],
                                        rowvar=False, bias=True) if len(points_for_cluster[x]) > 1 else
                                        np.identity(len(points_for_cluster[x][0])), signature='(),(m)->(n,n)')
        self.covariances_per_cluster = get_covariances(self.cluster_indexes, self.points_per_cluster)
        self.inv_covariances_per_cluster = np.vectorize(lambda x: np.linalg.pinv(x),
                                                        signature='(m,n)->(m,n)')(self.covariances_per_cluster)
