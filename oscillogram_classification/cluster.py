#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler

from training_data import TrainingData

SEED = 42
NUMBER_OF_CLUSTERS = 5  # for the battery voltage signal (sub-ROIs)
N_INIT = 50
MAX_ITER = 500
MAX_ITER_BARYCENTER = 500
RESAMPLING_DIVISOR = 100
INTERPOLATION_TARGET = "MIN"  # other options are 'MAX' and 'AVG'


def evaluate_performance_for_binary_clustering(y_train, y_pred):
    zero_neg_correct = 0
    assert set(np.unique(y_train)) == set(np.unique(y_pred))
    for i in range(len(y_train)):
        if y_train[i] == y_pred[i]:
            zero_neg_correct += 1
    print("---- correctly classified samples:")
    acc = (zero_neg_correct / len(y_train))
    print("if cluster 0 = NEG and cluster 1 = POS, then the accuracy is", acc)
    print("if cluster 0 = POS and cluster 1 = NEG, then the accuracy is", 1 - acc)
    print("...determine by visual comparison...")


def evaluate_performance(y_train, y_pred):
    assert set(np.unique(y_train)) == set(np.unique(y_pred))
    cluster_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    ground_truth_per_cluster = {0: [], 1: [], 2: [], 3: [], 4: []}

    for i in range(len(y_pred)):
        cluster_dict[y_pred[i]] += 1
        ground_truth_per_cluster[y_pred[i]].append(y_train[i])

    # ideal would be (6, 6, 6, 6, 6) - equally distributed
    print("cluster distribution:", cluster_dict.values())
    # each cluster should contain patches with identical labels, you don't know which one, but it must be identical
    print("ground truth per cluster:", ground_truth_per_cluster.values())


def plot_results(offset, title, clustering, x_train, y_train, y_pred, fig):
    print("#########################################################################################")
    print("results for", title)
    print("#########################################################################################")
    evaluate_performance(y_train, y_pred)
    for y in range(NUMBER_OF_CLUSTERS):
        ax = fig.add_subplot(3, NUMBER_OF_CLUSTERS, y + offset)
        for x in x_train[y_pred == y]:
            ax.plot(x.ravel(), "k-", alpha=.2)
        ax.plot(clustering.cluster_centers_[y].ravel(), "r-")
        ax.set_xlim(0, x_train.shape[1])
        ax.set_ylim(6, 15)
        ax.text(0.55, 0.85, 'Cluster %d' % y, transform=fig.gca().transAxes)
        if y == 0:
            plt.title(title)


def visualize_n_samples_per_class(x, y):
    """
    Iteratively visualizes one sample per class as long as the user enters '+'.

    :param x: sample series
    :param y: corresponding labels
    """
    classes = np.unique(y, axis=0)
    samples_by_class = {c: x[y == c] for c in classes}

    for sample in range(len(samples_by_class[classes[0]])):
        key = input("Enter '+' to see another sample per class\n")
        if key != "+":
            break
        fig1 = plt.figure()
        # create a single subplot that takes up the entire figure
        ax = fig1.add_subplot(1, 1, 1)
        for c in classes:
            if len(samples_by_class[c]) <= sample:
                print("no more complete sample distribution..")
                plt.close()
                return
            ax.plot(samples_by_class[c][sample], label="class " + str(c))
        ax.legend(loc="best")
        fig1.show()


def load_data():
    data = TrainingData(np.load("data/patch_data.npz", allow_pickle=True))
    visualize_n_samples_per_class(data[:][0], data[:][1])
    x_train = data[:][0]
    y_train = data[:][1]
    np.random.seed(SEED)
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    return x_train, y_train


def dir_path(path):
    """
    Returns path if it's valid, raises error otherwise.

    :param path: path to be checked
    :return: feasible path or error
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def create_dataset(norm, data_path):
    """
    Iterates through input data and generates an accumulated data set (.npz).

    :param norm: whether each sample should be normalized
    :param data_path: path to sample data
    """
    create_processed_time_series_dataset(data_path, norm)


def create_processed_time_series_dataset(data_path, norm):
    """
    Creates a processed time series dataset (.npz file containing all samples).

    :param data_path: path to sample data
    :param norm: whether each sample should be normalized
    """
    voltage_series = []
    labels = []
    for path in Path(data_path).glob('**/*.csv'):
        label, curr_voltages = read_oscilloscope_recording(path)
        labels.append(label)
        if norm:
            # TODO: experimentally compare different methods -> for now min-max worked best
            # curr_voltages = z_normalize_time_series(curr_voltages)
            # curr_voltages = min_max_normalize_time_series(curr_voltages)
            curr_voltages = decimal_scaling_normalize_time_series(curr_voltages, 2)
            # curr_voltages = logarithmic_normalize_time_series(curr_voltages, 10)
        voltage_series.append(curr_voltages)
    np.savez("data/patch_data.npz", np.array(voltage_series, dtype=object), np.array(labels))


def z_normalize_time_series(series):
    """
    Z-normalize the specified time series - 0 mean / 1 std_dev.

    :param series: time series to be normalized
    :return: normalized time series
    """
    return ((series - np.mean(series)) / np.std(series)).tolist()


def min_max_normalize_time_series(series):
    """
    Min-max-normalize the specified time series -> scale values to range [0, 1].

    :param series: time series to be normalized
    :return: normalized time series
    """
    minimum = np.min(series)
    maximum = np.max(series)
    return ((series - minimum) / (maximum - minimum)).tolist()


def decimal_scaling_normalize_time_series(series, power):
    """
    Decimal-scaling-normalize the specified time series -> largest absolute value < 1.0.

    :param series: time series to be normalized
    :param power: power used for scaling
    :return: normalized time series
    """
    return (np.array(series) / (10 ** power)).tolist()


def logarithmic_normalize_time_series(series, base):
    """
    Logarithmic-normalize the specified time series -> reduces impact of extreme values.

    :param series: time series to be normalized
    :param base: log base to be used
    :return: normalized time series
    """
    return (np.log(series) / np.log(base)).tolist()


def read_oscilloscope_recording(rec_file):
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


def zero_padding(patches):
    """
    Applies zero-padding to the provided patches and transforms them to the expected shape.

    :param patches: battery signal sub-ROI patches
    :return: padded / transformed patches
    """
    max_ts_length = max([len(patch) for patch in patches])
    padded_array = np.zeros((patches.shape[0], max_ts_length, 1))
    for i, ts in enumerate(patches):
        ts = np.array(ts).reshape(-1, 1)
        n_samples = ts.shape[0]
        padded_array[i, :n_samples, :] = ts
    return padded_array


def avg_padding(patches):
    """
    Applies average-padding to the provided patches and transforms them to the expected shape.

    :param patches: battery signal sub-ROI patches
    :return: padded / transformed patches
    """
    patches = patches.tolist()
    max_ts_length = max([len(patch) for patch in patches])
    for p in patches:
        avg_p = np.average(p)
        while len(p) < max_ts_length:
            p.append(avg_p)
    padded_array = np.array(patches)
    return padded_array.reshape((padded_array.shape[0], max_ts_length, 1))


def last_val_padding(patches):
    patches = patches.tolist()
    max_ts_length = max([len(patch) for patch in patches])
    for p in patches:
        while len(p) < max_ts_length:
            p.append(p[-1])
    padded_array = np.array(patches)
    return padded_array.reshape((padded_array.shape[0], max_ts_length, 1))


def periodic_padding(patches):
    patches = patches.tolist()
    max_ts_length = max([len(patch) for patch in patches])
    for p in patches:
        idx = 0
        while len(p) < max_ts_length:
            p.append(p[idx])
            idx += 1
    padded_array = np.array(patches)
    return padded_array.reshape((padded_array.shape[0], max_ts_length, 1))


def interpolation(patches):
    patches = patches.tolist()
    if INTERPOLATION_TARGET == "MIN":
        interpolation_target_len = min([len(patch) for patch in patches])
    elif INTERPOLATION_TARGET == "MAX":
        interpolation_target_len = max([len(patch) for patch in patches])
    elif INTERPOLATION_TARGET == "AVG":
        interpolation_target_len = int(np.average([len(patch) for patch in patches]))
    else:
        interpolation_target_len = min([len(patch) for patch in patches])

    for i in range(len(patches)):
        patches_arr = np.array(patches[i])
        patches_arr = patches_arr.reshape((1, len(patches[i]), 1))  # n_ts, sz, d
        patches[i] = TimeSeriesResampler(sz=interpolation_target_len).fit_transform(patches_arr).tolist()[0]
    padded_array = np.array(patches)
    return padded_array.reshape((padded_array.shape[0], interpolation_target_len, 1))


def preprocess_patches(patches):
    """
    Preprocesses the patches, i.e., performs padding and transforms them into a shape expected by `tslearn`.

    :param patches: battery signal sub-ROI patches
    :return: preprocessed patches
    """
    padded_array = interpolation(patches)
    print(padded_array.shape)
    print(padded_array[0])
    # TODO: do we really need this? usually way worse..
    # padded_array = TimeSeriesScalerMeanVariance().fit_transform(padded_array)
    return padded_array


if __name__ == '__main__':
    # input: raw oscilloscope data (one file per patch (sub ROI))
    # output: preprocessed data - one file containing data of all patches)
    parser = argparse.ArgumentParser(description='Clustering sub-ROI patches')
    parser.add_argument('--norm', action='store_true', help='normalize time series')
    parser.add_argument('--path', type=dir_path, required=True, help='path to the data to be processed')
    args = parser.parse_args()

    create_dataset(args.norm, args.path)
    x_train, y_train = load_data()
    x_train = preprocess_patches(x_train)

    print("original TS size:", len(x_train[0]))
    # # resample time series so that they reach the target size (sz - size of output TS)
    # #   -> we need to reduce the length of the TS (due to runtime, memory)
    x_train = TimeSeriesResampler(sz=len(x_train[0]) // RESAMPLING_DIVISOR).fit_transform(x_train)
    print("after down sampling:", len(x_train[0]))

    fig2 = plt.figure(figsize=(5 * NUMBER_OF_CLUSTERS, 3))

    print("Euclidean k-means")
    km = TimeSeriesKMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        verbose=True,
        random_state=SEED
    )
    y_pred = km.fit_predict(x_train)
    visualize_n_samples_per_class(x_train, y_pred)
    plot_results(1, "Euclidean $k$-means", km, x_train, y_train, y_pred, fig2)

    print("DBA k-means")
    dba_km = TimeSeriesKMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        metric="dtw",
        verbose=False,
        max_iter_barycenter=MAX_ITER_BARYCENTER,
        random_state=SEED
    )
    y_pred = dba_km.fit_predict(x_train)
    plot_results(1 + NUMBER_OF_CLUSTERS, "DBA $k$-means", dba_km, x_train, y_train, y_pred, fig2)
    visualize_n_samples_per_class(x_train, y_pred)

    print("Soft-DTW k-means")
    sdtw_km = TimeSeriesKMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        metric="softdtw",
        metric_params={"gamma": .01},
        verbose=True,
        max_iter_barycenter=MAX_ITER_BARYCENTER,
        random_state=SEED
    )
    y_pred = sdtw_km.fit_predict(x_train)
    plot_results(1 + 2 * NUMBER_OF_CLUSTERS, "Soft-DTW $k$-means", sdtw_km, x_train, y_train, y_pred, fig2)
    visualize_n_samples_per_class(x_train, y_pred)

    fig2.tight_layout()
    plt.show()
