import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.interpolate import spline
from Parser import parse_data
from Helpers import save_plot, plot_covariance
from ExpectationMaximization import ExpectationMaximization

data = parse_data()
em = ExpectationMaximization()
PLOT_MEAN_FOR_CLUSTERS_COUNT = True
PLOT_CLUSTERING_FOR_SOME_K = True
SAVE_PLOTS = True
VERBOSE = False

# Depending on the cluster centers that are chosen, results differ
# Though mostly between 3 and 4 clusters fit best for the current data set
if PLOT_MEAN_FOR_CLUSTERS_COUNT:
    cluster_count_experiments = [x for x in range(2, 20)]
    cluster_count_mean_distance_results = []

    for cluster_count in cluster_count_experiments:
        em.cluster(data, cluster_count)
        cluster_count_mean_distance_results.append(np.copy(em.mean_distance))

        if VERBOSE:
            for idx, points in enumerate(em.points_per_cluster):
                print('Points in cluster #{}: {}'.format(idx, len(points)))

    x = np.linspace(min(cluster_count_experiments), max(cluster_count_experiments), 300)
    y = spline(cluster_count_experiments, cluster_count_mean_distance_results, x)

    figure = plt.figure()
    plt.plot(x, y)
    plt.xlabel('Amount of clusters')
    plt.ylabel('Average Mahalanobis\' distance')

    if SAVE_PLOTS:
        save_plot(figure, './plots/avrg_distance_for_k.png')
    else:
        plt.show()

if PLOT_CLUSTERING_FOR_SOME_K:
    plot_for_k_s = [2,3,5,6, 20]
    # plot_for_k_s = [2]

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

            center = em.cluster_centers[cl_idx]
            covariance = em.covariances_per_cluster[cl_idx]
            plot_covariance(ax1, center[0], center[1], covariance)

        if SAVE_PLOTS:
            save_plot(fig, './plots/plot_for_k_{}.png'.format(k))
        else:
            plt.show()