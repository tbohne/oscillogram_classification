#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw, soft_dtw

from training_data import TrainingData

MODEL = "trained_models/dba_km.pkl"
DATA = "data/patch_data.npz"
METRIC = "DTW"
SEED = 42


def compute_distances(sample: np.ndarray, clustering_model: TimeSeriesKMeans) -> list:
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


def load_data() -> (np.ndarray, np.ndarray):
    """
    Loads the test samples.

    :return: test samples
    """
    data = TrainingData(np.load(DATA, allow_pickle=True))
    x_test = data[:][0]
    y_test = data[:][1]
    np.random.seed(SEED)
    idx = np.random.permutation(len(x_test))
    return x_test[idx], y_test[idx] if len(y_test) > 0 else []


def determine_best_matching_cluster_for_sample(sample: np.ndarray, clustering_model: TimeSeriesKMeans) -> int:
    """
    Determines the best matching cluster (smallest distance to centroid) for the specified sample.

    :param sample: sample to determine cluster for
    :param clustering_model: 'trained' clustering model
    :return: index of best-matching sample
    """
    distances = compute_distances(sample, clustering_model)
    # select the best-matching cluster for the new sample
    return int(np.argmin(distances))


def dir_path(path: str) -> str:
    """
    Returns path if it's valid, raises error otherwise.

    :param path: path to be checked
    :return: feasible path or error
    """
    if os.path.isfile(path) or os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def read_oscilloscope_recording(rec_file: Path) -> (int, list):
    """
    Reads the oscilloscope recording from the specified file.

    :param rec_file: oscilloscope recording file
    :return: list of voltage values (time series)
    """
    print("reading oscilloscope recording from", rec_file)
    label = None
    patches = ["patch0", "patch1", "patch2", "patch3", "patch4"]
    for patch in patches:
        if patch in str(rec_file).lower():
            label = int(patch[-1])
            break
    df = pd.read_csv(rec_file, delimiter=';', na_values=['-∞', '∞'])
    curr_voltages = list(df['Kanal A'].values)
    return label, curr_voltages


def create_processed_time_series_dataset(data_path: str) -> None:
    """
    Creates a processed time series dataset (.npz file containing all samples).

    :param data_path: path to sample data
    """
    voltage_series = []
    labels = []

    if os.path.isfile(data_path):
        label, curr_voltages = read_oscilloscope_recording(Path(data_path))
        labels.append(label)
        voltage_series.append(curr_voltages)

    for path in Path(data_path).glob('**/*.csv'):
        label, curr_voltages = read_oscilloscope_recording(path)
        labels.append(label)
        voltage_series.append(curr_voltages)

    np.savez(DATA, np.array(voltage_series, dtype=object), np.array(labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign new samples to predetermined clusters')
    parser.add_argument('--samples', type=dir_path, required=True, help='path to the samples to be assigned')
    args = parser.parse_args()
    create_processed_time_series_dataset(args.samples)

    # load saved clustering model from file
    model, y_pred, ground_truth = joblib.load(MODEL)
    test_x, test_y = load_data()

    for i in range(len(test_x)):
        test_sample = test_x[i]
        print("test sample excerpt:", test_sample[:15])
        best_matching_cluster = determine_best_matching_cluster_for_sample(test_sample, model)
        print("best matching cluster for new sample:", best_matching_cluster,
              "(", ground_truth[best_matching_cluster], ")")
        best_cluster = ground_truth[best_matching_cluster]

        # ground truth provided?
        if len(test_y) > 0:
            test_sample_ground_truth = test_y[i]
            print("ground truth:", test_sample_ground_truth)

            # if the ground truth matches the most prominent label in the cluster, it's a success
            d = {i: best_cluster.count(i) for i in np.unique(best_cluster)}
            most_prominent_entry = max(d, key=d.get)

            if test_sample_ground_truth == most_prominent_entry:
                print("SUCCESS: ground truth (", test_sample_ground_truth,
                      ") matches most prominent entry in cluster (", most_prominent_entry, ")")
            else:
                print("FAILURE: ground truth (", test_sample_ground_truth,
                      ") does not match most prominent entry in cluster (", most_prominent_entry, ")")
            print("-------------------------------------------------------------------------")
