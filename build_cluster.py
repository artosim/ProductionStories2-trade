# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans


seed = 0
numpy.random.seed(seed)

def cluster_viz(X_train, km, title):
    print(title)
    y_pred = km.fit_predict(X_train)
    plt.figure(figsize=(15, 15))
    for yi in range(km.n_clusters):
        plt.subplot(3, 3, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)
    plt.tight_layout()
    plt.show()

def cluster_kernel_viz(X_train, km, title):
    print(title)
    y_pred = km.fit_predict(X_train)
    plt.figure(figsize=(15, 15))
    for yi in range(km.n_clusters):
        plt.subplot(3, 3, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)
    plt.tight_layout()
    plt.show()


def cluster_plot_kmeans(X_train, n_clusters, metric, title, seed=0):
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=seed)
    cluster_viz(X_train, km, title)

def cluster_plot_kernel_kmeans(X_train, n_clusters, kernel, title, seed=0):
    km = KernelKMeans(n_clusters=n_clusters, kernel=kernel, random_state=seed)
    cluster_kernel_viz(X_train, km, title)