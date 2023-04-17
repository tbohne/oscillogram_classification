#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import joblib
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw, soft_dtw

from training_data import TrainingData

MODEL = "dba_km.pkl"
DATA = "data/patch_data.npz"
METRIC = "DTW"
SEED = 42
TEST_SAMPLE_CNT = 5


def compute_distances(sample: list, clustering_model: TimeSeriesKMeans) -> list:
    """
    Computes the distance between the provided sample and each cluster of the specified model using the configured
    metric.

    :param sample: new sample to be assigned to the closest cluster
    :param clustering_model: "trained" clustering model
    :return: computed cluster distances
    """
    if METRIC == "DTW":
        return [dtw(sample, cluster_centroid) for cluster_centroid in clustering_model.cluster_centers_]
    elif METRIC == "SOFT_DTW":
        return [soft_dtw(sample, cluster_centroid) for cluster_centroid in clustering_model.cluster_centers_]
    else:
        # default option is DTW
        return [dtw(sample, cluster_centroid) for cluster_centroid in clustering_model.cluster_centers_]


def set_up_predefined_clusters(predictions, ground_truth):
    """
    Sets up the predefined clusters.

    :param predictions: predictions of the predefined clustering model for the 'training data'
    :param ground_truth: ground truth labels for the 'training data'
    :return: ground truth values for each cluster
    """
    # we have 5 clusters, i.e., 5 sub-ROIs
    cluster_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    ground_truth_per_cluster = {0: [], 1: [], 2: [], 3: [], 4: []}

    for i in range(len(predictions)):
        cluster_dict[predictions[i]] += 1
        ground_truth_per_cluster[predictions[i]].append(ground_truth[i])
    return ground_truth_per_cluster


def load_data():
    """
    Loads the test samples.

    :return: test samples
    """
    data = TrainingData(np.load(DATA, allow_pickle=True))
    x_train = data[:][0]
    y_train = data[:][1]
    np.random.seed(SEED)
    idx = np.random.permutation(len(x_train))
    return x_train[idx], y_train[idx]


if __name__ == '__main__':
    # load saved clustering model from file
    model, y_pred = joblib.load(MODEL)
    train_x, train_y = load_data()
    predefined_clusters = set_up_predefined_clusters(y_pred, train_y)

    for i in range(TEST_SAMPLE_CNT):
        test_sample = train_x[i * 3]
        test_sample_ground_truth = train_y[i * 3]
        print("test sample excerpt:", test_sample[:15])
        print("ground truth:", test_sample_ground_truth)

        distances = compute_distances(test_sample, model)

        # select the best-matching cluster for the new sample
        best_matching_cluster = int(np.argmin(distances))
        print("best matching cluster for new sample:", best_matching_cluster,
              "(", predefined_clusters[best_matching_cluster], ")")
        best_cluster = predefined_clusters[best_matching_cluster]

        # if the ground truth matches the most prominent label in the cluster, it's a success
        d = {i: best_cluster.count(i) for i in np.unique(best_cluster)}
        most_prominent_entry = max(d, key=d.get)

        if test_sample_ground_truth == most_prominent_entry:
            print("SUCCESS: ground truth (", test_sample_ground_truth, ") matches most prominent entry in cluster (",
                  most_prominent_entry, ")")
        else:
            print("FAILURE: ground truth (", test_sample_ground_truth,
                  ") does not match most prominent entry in cluster (", most_prominent_entry, ")")
        print("-------------------------------------------------------------------------")
