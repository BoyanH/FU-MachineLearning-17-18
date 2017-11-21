from Parser import parse_data
import numpy as np
from Helpers import save_plot
from random import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import spline
import random

CHOOSE_INITIAL_CENTERS_RANDOMLY = True
USE_DISTANCE_THRESHOLD = True
MOVEMENT_THRESHOLD = 0.002
DISTANCE_THRESHOLD = 6

VERBOSE = True
PLOT_MEAN_FOR_CLUSTERS_COUNT = False
PLOT_CLUSTERING_FOR_SOME_K = True
SAVE_PLOTS = True


class ExpectationMaximization:
    @staticmethod
    def get_initial_centers_from_data_set(data, k):
        if CHOOSE_INITIAL_CENTERS_RANDOMLY:
            seed = random.randint(0, 1000)
            random.seed(8)
            return np.array([x for x in random.choices(data, k=k)])

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
        self.covariances_per_cluster = []
        self.cluster_indexes = None
        self.last_diff = None
        self.mean_distance = None
        self.max_iterations = None

    def reset_points_per_cluster(self):
        self.points_per_cluster = [[x] for x in self.cluster_centers]

    def cluster(self, data, k_clusters, max_iterations=30):
        self.data = data
        self.k_clusters = k_clusters
        self.cluster_indexes = [x for x in range(self.k_clusters)]
        self.cluster_centers = ExpectationMaximization.get_initial_centers_from_data_set(data, k_clusters)
        self.covariances_per_cluster = [[np.identity(len(x))] for x in self.cluster_centers]
        self.reset_points_per_cluster()
        self.max_iterations = max_iterations
        self.apply_expectation_maximization()

    def apply_expectation_maximization(self, k=0):
        if VERBOSE:
            print('iteration: {}'.format(k))
        old_centers = np.copy(self.cluster_centers)
        old_distance = self.mean_distance
        self.reset_points_per_cluster()
        self.assign_points_to_clusters()
        self.calculate_cluster_centers()
        self.calculate_covariances()
        self.update_mean_distance_to_cluster_centers()

        if USE_DISTANCE_THRESHOLD:
            if (self.mean_distance > DISTANCE_THRESHOLD and
                        k < self.max_iterations and (old_distance is None or old_distance > self.mean_distance)):
                self.apply_expectation_maximization(k + 1)
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

        self.mean_distance = np.array(flattened_distances).mean()

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
        return (x - self.cluster_centers[i_cluster]).dot(
            self.covariances_per_cluster[i_cluster]).dot((x - self.cluster_centers[i_cluster]).T)

    def calculate_cluster_centers(self):
        get_cluster_centers = np.vectorize(lambda x, points_per_center: points_per_center[x].mean(0),
                                           signature='(),(m)->(n)')
        self.cluster_centers = get_cluster_centers(self.cluster_indexes, self.points_per_cluster)

    def calculate_covariances(self):
        # if we only have the center in our cluster (empty cluster) then just use the identity
        # matrix as covariance matrix again
        get_covariances = np.vectorize(lambda x, points: np.linalg.pinv(np.cov(points[x],
                                                                rowvar=False, bias=True)) if len(points[x]) > 1 else
        np.identity(len(points[x][0])), signature='(),(m)->(n,n)')
        self.covariances_per_cluster = get_covariances(self.cluster_indexes, self.points_per_cluster)

    def get_average_distance(self):
        pass


data = parse_data()
em = ExpectationMaximization()

# Depending on the cluster centers that are chosen, results differ
# Though mostly between 3 and 4 clusters fit best for the current data set
if PLOT_MEAN_FOR_CLUSTERS_COUNT:
    cluster_count_experiments = [x for x in range(2, 8)]
    cluster_count_mean_distance_results = []

    for cluster_count in cluster_count_experiments:
        em.cluster(data, cluster_count)
        cluster_count_mean_distance_results.append(em.mean_distance)

        if VERBOSE:
            for idx, points in enumerate(em.points_per_cluster):
                print('Points in cluster #{}: {}'.format(idx, len(points)))

    x = np.linspace(min(cluster_count_experiments), max(cluster_count_experiments), 300)
    y = spline(cluster_count_experiments, cluster_count_mean_distance_results, x)

    plt.plot(x, y)
    plt.xlabel('Amount of clusters')
    plt.ylabel('Average Mahalanobis\' distance')
    plt.show()

if PLOT_CLUSTERING_FOR_SOME_K:
    plot_for_k_s = [2,3,5,6]

    for k in plot_for_k_s:
        em.cluster(data, k)
        colors = cm.rainbow(np.linspace(0, 1, k))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for cl_idx in em.cluster_indexes:
            X = em.points_per_cluster[cl_idx]
            x, y = zip(*X)
            # ax1.figure(figsize=(15, 10))
            ax1.scatter(x, y, edgecolors="black", c=colors[cl_idx])

        if SAVE_PLOTS:
            save_plot(fig, './plots/plot_for_k_{}.png'.format(k))
        else:
            plt.show()
