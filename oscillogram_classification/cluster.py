#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

from training_data import TrainingData

SEED = 42
NUMBER_OF_CLUSTERS = 2  # atm just POS and NEG


def plot_results(offset, title, clustering, x_train, y_train, y_pred):
    print("#########################################################################################")
    print("results for", title)
    print("#########################################################################################")
    for y in range(2):
        plt.subplot(3, 2, y + offset)
        for x in x_train[y_pred == y]:
            plt.plot(x.ravel(), "k-", alpha=.2)
        plt.plot(clustering.cluster_centers_[y].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85, 'Cluster %d' % y, transform=plt.gca().transAxes)
        if y == 0:
            plt.title(title)


np.random.seed(SEED)
data = TrainingData(np.load("data/MILESTONE_DEMO/training_data.npz", allow_pickle=True))
x_train = (data[:][0])[..., np.newaxis]
y_train = (data[:][1])[..., np.newaxis]

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

x_train = TimeSeriesScalerMeanVariance().fit_transform(x_train)

# we need to reduce the length of the TS (due to runtime)
# TODO: determine feasible size via experiments
x_train = TimeSeriesResampler(sz=len(x_train) // 4).fit_transform(x_train)
sz = x_train.shape[1]
plt.figure()

print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=NUMBER_OF_CLUSTERS, verbose=False, random_state=SEED)
y_pred = km.fit_predict(x_train)
plot_results(1, "Euclidean $k$-means", km, x_train, y_train, y_pred)

print("DBA k-means")
dba_km = TimeSeriesKMeans(
    n_clusters=NUMBER_OF_CLUSTERS, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=50, random_state=SEED)
y_pred = dba_km.fit_predict(x_train)
plot_results(3, "DBA $k$-means", dba_km, x_train, y_train, y_pred)

print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(
    n_clusters=NUMBER_OF_CLUSTERS, metric="softdtw", metric_params={"gamma": .01}, verbose=False, random_state=SEED)
y_pred = sdtw_km.fit_predict(x_train)
plot_results(5, "Soft-DTW $k$-means", sdtw_km, x_train, y_train, y_pred)

plt.tight_layout()
plt.show()
