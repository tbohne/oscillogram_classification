#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler

from training_data import TrainingData

SEED = 42
NUMBER_OF_CLUSTERS = 7  # for the battery voltage signal (sub-ROIs)
N_LABEL = 5  # number of integers that appear as labels in the file names of the patches
N_INIT = 50
MAX_ITER = 500
MAX_ITER_BARYCENTER = 500
RESAMPLING_DIVISOR = 10
INTERPOLATION_TARGET = "MIN"  # other options are 'MAX' and 'AVG'
SMALL_VAL = 0.0000001


def evaluate_performance_for_binary_clustering(ground_truth: np.ndarray, predictions: np.ndarray) -> None:
    """
    Evaluates the performance of binary clustering, i.e., compares the ground truth labels to the predictions.

    :param ground_truth: ground truth labels
    :param predictions: predicted labels (clusters)
    """
    zero_neg_correct = 0
    assert set(np.unique(ground_truth)) == set(np.unique(predictions))
    for i in range(len(ground_truth)):
        if ground_truth[i] == predictions[i]:
            zero_neg_correct += 1
    print("---- correctly classified samples:")
    acc = (zero_neg_correct / len(ground_truth))
    print("if cluster 0 = NEG and cluster 1 = POS, then the accuracy is", acc)
    print("if cluster 0 = POS and cluster 1 = NEG, then the accuracy is", 1 - acc)
    print("...determine by visual comparison...")


def evaluate_performance(ground_truth: np.ndarray, predictions: np.ndarray) -> dict:
    """
    Evaluates the clustering performance for battery signals. An ROI detection algorithm provides the input for the
    clustering of the cropped sub-ROIs. The ROIs are divided into five categories for the battery signals. The five
    categories (clusters) are practically motivated, based on semantically meaningful regions that an expert would look
    at when searching for anomalies.

    :param ground_truth: ground truth labels
    :param predictions: predicted labels (clusters)
    :return: ground truth dictionary
    """

    # TODO: not necessarily a good assumption: there can be more than one cluster per patch type
    # assert set(np.unique(ground_truth)) == set(np.unique(predictions))
    # we have $k$ clusters, i.e., $k$ sub-ROIs
    cluster_dict = {i: 0 for i in range(NUMBER_OF_CLUSTERS)}
    ground_truth_per_cluster = {i: [] for i in range(NUMBER_OF_CLUSTERS)}

    for i in range(len(predictions)):
        cluster_dict[predictions[i]] += 1
        ground_truth_per_cluster[predictions[i]].append(ground_truth[i])

    # ideal would be (n, n, n, n, n) - equally distributed
    print("cluster distribution:", list(cluster_dict.values()))

    # each cluster should contain patches with identical labels, you don't know which one, but it must be identical
    print("ground truth per cluster:")
    for val in ground_truth_per_cluster.values():
        print("\t-", val, "\n")
    return ground_truth_per_cluster


def plot_results(offset: int, title: str, clustering: TimeSeriesKMeans, train_x: np.ndarray, train_y: np.ndarray,
                 pred_y: np.ndarray, fig: plt.Figure) -> dict:
    """
    Plots the results of the clustering procedure.

    :param offset: y-offset for the data to be displayed
    :param title: plot title
    :param clustering: clustering results
    :param train_x: patches that were clustered
    :param train_y: ground truth of clustered patches
    :param pred_y: predictions (cluster assignments)
    :param fig: figure to add plot to
    :return: ground truth dictionary
    """
    print("#########################################################################################")
    print("results for", title)
    print("#########################################################################################")
    ground_truth_dict = evaluate_performance(train_y, pred_y)
    for y in range(NUMBER_OF_CLUSTERS):
        ax = fig.add_subplot(3, NUMBER_OF_CLUSTERS, y + offset)
        for x in train_x[pred_y == y]:
            ax.plot(x.ravel(), "k-", alpha=.2)
        ax.plot(clustering.cluster_centers_[y].ravel(), "r-")
        ax.set_xlim(0, train_x.shape[1])
        ax.set_ylim(6, 15)
        ax.text(0.55, 0.85, 'Cluster %d' % y, transform=fig.gca().transAxes)
        if y == 0:
            plt.title(title)
    return ground_truth_dict


def visualize_n_samples_per_class(x: np.ndarray, y: np.ndarray) -> None:
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
        sample_fig = plt.figure()
        # create a single subplot that takes up the entire figure
        ax = sample_fig.add_subplot(1, 1, 1)
        for c in classes:
            if len(samples_by_class[c]) <= sample:
                print("no more complete sample distribution..")
                plt.close()
                return
            ax.plot(samples_by_class[c][sample], label="class " + str(c))
        ax.legend(loc="best")
        sample_fig.show()


def load_data() -> (np.ndarray, np.ndarray):
    """
    Loads the data to be clustered.

    :return: (patches to be clustered, ground truth labels)
    """
    data = TrainingData(np.load("data/patch_data.npz", allow_pickle=True))
    visualize_n_samples_per_class(data[:][0], data[:][1])
    train_x = data[:][0]
    train_y = data[:][1]
    np.random.seed(SEED)
    idx = np.random.permutation(len(train_x))
    train_x = train_x[idx]
    train_y = train_y[idx]
    return train_x, train_y


def dir_path(path: str) -> str:
    """
    Returns path if it's valid, raises error otherwise.

    :param path: path to be checked
    :return: feasible path or error
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def create_dataset(norm: bool, data_path: str) -> None:
    """
    Iterates through input data and generates an accumulated data set (.npz).

    :param norm: whether each sample should be normalized
    :param data_path: path to sample data
    """
    create_processed_time_series_dataset(data_path, norm)


def clean_incorrect_patches(paths: list) -> list:
    """
    Removes time series that were not split into the expected number of patches (either too many or too little).

    5 patches are expected for positive measurements and 3 for negative ones.

    :param paths: list of paths to all patches. If negative patches are included, their path should contain the word "negative".
    :return: list of paths to all patches of time series that have the expected number of patches
    """
    cleaned_paths = []
    for path_object in paths:
        path = str(path_object)
        measurement_id = path.split(os.path.sep)[-1].split("_")[0]
        if "negative" in path:
            if not any(measurement_id + "_patch3" in str(other_path) and "negative" in str(other_path) for other_path in
                       paths) and any(
                measurement_id + "_patch2" in str(other_path) and "negative" in str(other_path) for other_path in
                paths):
                cleaned_paths.append(path_object)
        else:
            if not any(
                    measurement_id + "_patch5" in str(other_path) and not "negative" in str(other_path) for other_path
                    in paths) and any(
                measurement_id + "_patch4" in str(other_path) and not "negative" in str(other_path) for other_path
                in paths):
                cleaned_paths.append(path_object)
    return cleaned_paths


def create_processed_time_series_dataset(data_path: str, norm: bool = False) -> None:
    """
    Creates a processed time series dataset (.npz file containing all samples).

    :param data_path: path to sample data
    :param norm: whether each sample should be normalized
    """
    voltage_series = []
    labels = []
    measurement_ids = []

    if os.path.isfile(data_path):
        paths = list(Path(data_path))
    else:
        paths = list(Path(data_path).glob('**/*.csv'))
        paths = clean_incorrect_patches(paths)
    for path in paths:
        label, curr_voltages = read_oscilloscope_recording(path)
        labels.append(label)
        if norm:
            # TODO: experimentally compare different methods -> for now, decimal worked best
            # curr_voltages = z_normalize_time_series(curr_voltages)
            # curr_voltages = min_max_normalize_time_series(curr_voltages)
            curr_voltages = decimal_scaling_normalize_time_series(curr_voltages, 2)
            # curr_voltages = logarithmic_normalize_time_series(curr_voltages, 10)
        voltage_series.append(curr_voltages)
        measurement_id = str(path).split(os.path.sep)[-1].split("_")[0]
        measurement_ids.append(measurement_id + "_negative") if "negative" in str(path) else measurement_ids.append(
            measurement_id + "_positive")

    np.savez("data/patch_data.npz", np.array(voltage_series, dtype=object), np.array(labels))
    np.savetxt("data/patch_measurement_ids.csv", measurement_ids, delimiter=',', fmt='%s')


def z_normalize_time_series(series: list) -> list:
    """
    Z-normalize the specified time series - 0 mean / 1 std_dev.

    :param series: time series to be normalized
    :return: normalized time series
    """
    std_dev = np.std(series)
    if std_dev == 0.0:
        std_dev = SMALL_VAL  # value not important, just prevent division by zero
        # (x - mean) is 0 anyway when the standard deviation is 0 -> 0 in the end
    return ((series - np.mean(series)) / std_dev).tolist()


def min_max_normalize_time_series(series: list) -> list:
    """
    Min-max-normalize the specified time series -> scale values to range [0, 1].

    :param series: time series to be normalized
    :return: normalized time series
    """
    minimum = np.min(series)
    maximum = np.max(series)
    denominator = maximum - minimum
    if denominator == 0.0:
        denominator = SMALL_VAL
        # if (max-min) is 0, they are equal, which means that all values are the same,
        # but then the numerator is 0 anyway
    return ((series - minimum) / denominator).tolist()


def decimal_scaling_normalize_time_series(series: list, power: int) -> list:
    """
    Decimal-scaling-normalize the specified time series -> largest absolute value < 1.0.

    :param series: time series to be normalized
    :param power: power used for scaling
    :return: normalized time series
    """
    return (np.array(series) / (10 ** power)).tolist()


def logarithmic_normalize_time_series(series: list, base: int) -> list:
    """
    Logarithmic-normalize the specified time series -> reduces impact of extreme values.

    :param series: time series to be normalized
    :param base: log base to be used
    :return: normalized time series
    """
    return (np.log(series) / np.log(base)).tolist()


def read_oscilloscope_recording(rec_file: Path) -> (int, list):
    """
    Reads the oscilloscope recording from the specified file.

    :param rec_file: oscilloscope recording file
    :return: (ground truth label, list of voltage values (time series))
    """
    print("reading oscilloscope recording from", rec_file)
    label = None
    patches = ["patch" + str(i) for i in range(N_LABEL)]

    for patch in patches:
        if patch in str(rec_file).lower():
            label = int(patch[-1])
            # create 7 label classes: patch types 1-4 occurring in positive measurements, patch types 5-6 occurring in
            # negative measurements, and patch type 0 that appears in both positive and negative measurements
            if "negative" in str(rec_file).lower() and label != 0:
                label += len(patches) - 1
            break

    df = pd.read_csv(rec_file, delimiter=';', na_values=['-∞', '∞'])
    curr_voltages = list(df['Kanal A'].values)
    return label, curr_voltages


def zero_padding(patches: np.ndarray) -> np.ndarray:
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


def avg_padding(patches: np.ndarray) -> np.ndarray:
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


def last_val_padding(patches: np.ndarray) -> np.ndarray:
    """
    Applies last-value-padding to the provided patches and transforms them to the expected shape.

    :param patches: battery signal sub-ROI patches
    :return: padded / transformed patches
    """
    patches = patches.tolist()
    max_ts_length = max([len(patch) for patch in patches])
    for p in patches:
        while len(p) < max_ts_length:
            p.append(p[-1])
    padded_array = np.array(patches)
    return padded_array.reshape((padded_array.shape[0], max_ts_length, 1))


def periodic_padding(patches: np.ndarray) -> np.ndarray:
    """
    Applies periodic-padding to the provided patches and transforms them to the expected shape.

    :param patches: battery signal sub-ROI patches
    :return: padded / transformed patches
    """
    patches = patches.tolist()
    max_ts_length = max([len(patch) for patch in patches])
    for p in patches:
        idx = 0
        while len(p) < max_ts_length:
            p.append(p[idx])
            idx += 1
    padded_array = np.array(patches)
    return padded_array.reshape((padded_array.shape[0], max_ts_length, 1))


def interpolation(patches: np.ndarray) -> np.ndarray:
    """
    Resamples the provided patches and transforms them to the expected shape.

    :param patches: battery signal sub-ROI patches
    :return: padded  / transformed patches
    """
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


def preprocess_patches(patches: np.ndarray) -> np.ndarray:
    """
    Preprocesses the patches, i.e., performs padding and transforms them into a shape expected by `tslearn`.

    :param patches: battery signal sub-ROI patches
    :return: preprocessed patches
    """
    padded_array = interpolation(patches)
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
    # bring all patches to the same length
    x_train = preprocess_patches(x_train)

    print("original TS size:", len(x_train[0]))
    # # resample time series so that they reach the target size (sz - size of output TS)
    # #   -> we need to reduce the length of the TS (due to runtime, memory)
    x_train = TimeSeriesResampler(sz=len(x_train[0]) // RESAMPLING_DIVISOR).fit_transform(x_train)
    print("after down sampling:", len(x_train[0]))

    fig2 = plt.figure(figsize=(5 * NUMBER_OF_CLUSTERS, 3))

    print("Euclidean k-means")
    euclidean_km = TimeSeriesKMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        verbose=True,
        random_state=SEED
    )
    y_pred = euclidean_km.fit_predict(x_train)
    visualize_n_samples_per_class(x_train, y_pred)
    ground_truth = plot_results(1, "Euclidean $k$-means", euclidean_km, x_train, y_train, y_pred, fig2)
    joblib.dump((euclidean_km, y_pred, ground_truth), 'trained_models/euclidean_km.pkl')  # save model to file

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
    visualize_n_samples_per_class(x_train, y_pred)
    ground_truth = plot_results(1 + NUMBER_OF_CLUSTERS, "DBA $k$-means", dba_km, x_train, y_train, y_pred, fig2)
    joblib.dump((dba_km, y_pred, ground_truth), 'trained_models/dba_km.pkl')  # save model to file

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
    visualize_n_samples_per_class(x_train, y_pred)
    ground_truth = plot_results(1 + 2 * NUMBER_OF_CLUSTERS, "Soft-DTW $k$-means", sdtw_km, x_train, y_train, y_pred,
                                fig2)
    joblib.dump((sdtw_km, y_pred, ground_truth), 'trained_models/sdtw_km.pkl')  # save model to file

    fig2.tight_layout()
    plt.show()
