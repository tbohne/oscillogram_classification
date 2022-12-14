#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
import uuid
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh import select_features
from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute


def read_oscilloscope_recording(rec_file):
    """
    Reads the oscilloscope recording from the specified file in standard format.

    :param rec_file: oscilloscope recording file
    :return: label, list of voltage values (time series)
    """
    print("reading oscilloscope recording from", rec_file)
    # check whether it's a labeled file
    if "pos" in str(rec_file).lower() or "neg" in str(rec_file).lower():
        label = 1 if "pos" in str(rec_file).lower() else 0  # label: pos (1) / neg (0)
    else:
        label = None

    df = pd.read_csv(rec_file, delimiter=';', na_values=['-∞', '∞'])
    df = df[1:].apply(lambda x: x.str.replace(',', '.')).astype(float).dropna()
    curr_voltages = list(df['Kanal A'].values)
    return label, curr_voltages


def read_voltage_only_format_recording(rec_file):
    """
    Reads the oscilloscope recording from the specified file in voltage-only format.

    :param rec_file: oscilloscope recording file
    :return: label, list of voltage values (time series)
    """
    print("reading oscilloscope recording from", rec_file)
    assert "pos" in str(rec_file).lower() or "neg" in str(rec_file).lower()
    # label: pos (1) / neg (0)
    label = 1 if "pos" in str(rec_file).lower() else 0
    a = pd.read_csv(rec_file, delimiter=",", na_values=["-∞", "∞"], names=["Kanal A"])
    curr_voltages = list(a['Kanal A'].values)
    return label, curr_voltages


def equalize_sample_sizes(voltage_series):
    """
    Naive method to equalize sample sizes - reduce all samples from the end to the size of the smallest sample.

    :param voltage_series: list of samples
    """
    min_size = min([len(sample) for sample in voltage_series])
    # reduce all samples with too many data points
    for i in range(len(voltage_series)):
        if len(voltage_series[i]) > min_size:
            remove = len(voltage_series[i]) - min_size
            voltage_series[i] = voltage_series[i][: len(voltage_series[i]) - remove]


def z_normalize_time_series(series):
    """
    Z-normalize the specified time series - 0 mean / 1 std_dev.

    :param series: time series to be z-normalized
    :return: normalized time series
    """
    return (series - np.mean(series)) / np.std(series)


def dask_feature_extraction_for_large_input_data(dataframe, partitions, on_chunk=True, simple_return=True):
    """
    Performs feature extraction / selection with tsfresh using Dask dataframes.

        "Dask dataframes allow you to scale your computation beyond your local memory (via partitioning
         the data internally) and even to large clusters of machines."

    :param dataframe: pandas dataframe to perform feature extraction on
    :param partitions: number of partitions to be generated in the Dask dataframe
    :param on_chunk: perform feature extraction on chunk
    :param simple_return: simply returns the features (no further processing)
    :return: extracted / selected features
    """
    # convert pandas df to dask df
    df = dd.from_pandas(dataframe, npartitions=partitions)
    print(df.head())

    if on_chunk:
        df_grouped = df.groupby(["id"])
        print(df_grouped)
        print("extract features..")
        features = dask_feature_extraction_on_chunk(df_grouped,
                                                    column_id="id", column_kind="id", column_sort="Zeit",
                                                    column_value="Kanal A",
                                                    default_fc_parameters=ComprehensiveFCParameters())

        if simple_return:
            print("ext. features:")
            print(features)
            # TODO: runs out of memory when computing the pandas dataframe (compute=True by default)
            print(features.head(npartitions=-1, compute=False))
            print(features.columns)
        else:
            # TODO: causes OOB error
            features = features.categorize(columns=["variable"])
            features = features.reset_index(drop=True)
            feature_table = features.pivot_table(index="id", columns="variable", values="value", aggfunc="sum")
            print(feature_table)
    else:
        print("extract features..")
        features = extract_features(df, column_id="id", column_sort="Zeit", pivot=False)
        print("ext. features:")
        print(features)
        print(features.head(compute=False))
        # TODO: runs out of memory when computing the pandas dataframe
        result = features.compute()
        print(result)

    # TODO: takes too much time / memory
    print(features.head())
    return features


def pandas_feature_extraction(df, labels):
    """
    Performs feature extraction / selection using tsfresh.

    :param df: pandas dataframe (set of time series)
    :param labels: corresponding labels
    :return: extracted / selected features
    """
    print("ext. features..")
    # n_jobs = 0 -> no parallelization -> reduces memory consumption (however, memory consumption still too high)
    # using EfficientFCParameters to reduce memory consumption, ignoring features that are too computationally expensive
    return extract_relevant_features(
        df, labels, column_id="id", column_sort="Zeit", default_fc_parameters=EfficientFCParameters()
    )


def pandas_feature_extraction_manual(df, labels, data_type):
    """
    Performs feature extraction / selection using tsfresh (all steps manually).

    :param df: pandas dataframe (set of time series)
    :param labels: corresponding labels
    :param data_type: train | test | validation
    :return: (filtered features, all extracted features)
    """
    print("ext. features..")
    # uses EfficientFCParameters to make it feasible on my machine (RAM-wise)
    extracted_features = extract_features(
        df, column_id="id", column_sort="Zeit", default_fc_parameters=EfficientFCParameters()
    )
    print("impute..")
    impute(extracted_features)

    filtered_features = None
    # filtering is based on statistics and requires at least more than 1 sample
    if len(labels) > 1:
        print("select relevant features..")
        filtered_features = select_features(extracted_features, labels)
        print("selected features:")
        print(filtered_features)
        filtered_features.to_csv('data/%s_filtered_features.csv' % data_type, encoding='utf-8', index=False)

    print("saving to csv..")
    extracted_features.to_csv('data/%s_complete_features.csv' % data_type, encoding='utf-8', index=False)

    # print("settings:")
    # kind_to_fc_params = tsfresh.feature_extraction.settings.from_columns(filtered_features)
    # print(kind_to_fc_params)

    return filtered_features, extracted_features


def create_processed_time_series_dataset(data_path, diff_format, z_norm, data_type):
    """
    Creates a processed time series dataset (.npz file containing all samples).

    :param data_path: path to sample data
    :param diff_format: whether the samples are provided in a differing format (see above)
    :param z_norm: whether each sample should be z-normalized
    :param data_type: train | test | validation
    """
    labels = []
    voltage_series = []
    for path in Path(data_path).glob('**/*.csv'):
        label, curr_voltages = read_oscilloscope_recording(
            path) if not diff_format else read_voltage_only_format_recording(path)
        labels.append(label)
        if z_norm:
            curr_voltages = z_normalize_time_series(curr_voltages)
        voltage_series.append(curr_voltages)
    equalize_sample_sizes(voltage_series)
    np.savez("data/%s_data.npz" % data_type, np.array(voltage_series, dtype=object), np.array(labels))


def filter_extracted_features_based_on_list(feature_list, all_extracted_features):
    """
    Filters the extracted features based on the specified features list.
    Aim: E.g. to align the test / validation set with the training set (use same features).

    :param feature_list: list of features to be considered
    :param all_extracted_features: extracted features to be filtered
    :return: filtered features (aligned with the specified feature list)
    """
    to_drop = [feature for feature in all_extracted_features.columns if feature not in feature_list]
    filtered_features = all_extracted_features.drop(columns=to_drop)

    # establish same order of features as in feature list (e.g. as in training set)
    filtered_features = filtered_features.reindex(columns=feature_list)

    assert len(filtered_features.columns.to_numpy()) == len(feature_list)
    return filtered_features


def create_feature_vector_dataset(data_path, data_type, feature_list):
    """
    Creates a dataset based on extracted / selected features of the time series.

    :param data_path: path to sample data
    :param data_type: train | test | validation
    :param feature_list: list of features to be considered (in case of feature extraction)
    """
    print("preparing feature extraction..")
    df = None
    labels = {}
    for path in Path(data_path).glob('**/*.csv'):
        print("reading oscilloscope recording from", path)
        assert "pos" in str(path).lower() or "neg" in str(path).lower()
        # label: pos (1) / neg (0)
        label = 1 if "pos" in str(path).lower() else 0

        curr_df = pd.read_csv(path, delimiter=';', na_values=['-∞', '∞'])
        curr_df = curr_df[1:].apply(lambda x: x.str.replace(',', '.')).astype(float).dropna()

        unique_id = uuid.uuid4().hex
        curr_df.insert(0, "id", [unique_id for _ in range(len(curr_df))], True)
        labels[unique_id] = label

        if df is None:
            df = curr_df
        else:
            df = pd.concat([df, curr_df], ignore_index=True)

    print(df)
    labels = pd.Series(labels)
    print(labels)

    # features = dask_feature_extraction_for_large_input_data(df, 4, on_chunk=False, simple_return=True)
    # features = pandas_feature_extraction(df, labels)
    filtered_features, all_extracted_features = pandas_feature_extraction_manual(df, labels, data_type)

    # filtered features only provided when sample size is large enough
    if filtered_features is not None:
        print("creating filtered feature vectors generated by tsfresh relevance estimation..")
        print("number of features:", np.array(filtered_features.columns))
        print(filtered_features.head())
        # filtered feature vectors generated by tsfresh relevance estimation
        filtered_feature_columns = np.array(filtered_features.columns)
        res_feature_vectors = filtered_features.to_numpy()
        np.savez("data/%s_filtered_feature_vectors.npz" % data_type,
                 res_feature_vectors, labels.to_numpy(), filtered_feature_columns)

    # complete feature vectors
    print("creating complete feature vectors (using all features considered by tsfresh)..")
    complete_feature_columns = np.array(all_extracted_features.columns)
    print("number of features:", len(complete_feature_columns))
    res_complete_feature_vectors = all_extracted_features.to_numpy()
    np.savez("data/%s_complete_feature_vectors.npz" % data_type,
             res_complete_feature_vectors, labels.to_numpy(), complete_feature_columns)

    if feature_list is not None:
        # filtered feature vectors by manual alignment with provided feature list
        print("creating manually filtered feature vectors based on provided csv list..")
        manually_filtered_features = filter_extracted_features_based_on_list(feature_list, all_extracted_features)
        filtered_feature_columns = np.array(manually_filtered_features.columns)
        print("number of features:", len(filtered_feature_columns))
        manually_filtered_features.to_csv(
            'data/%s_manually_filtered_features.csv' % data_type, encoding='utf-8', index=False)
        manually_filtered_features = manually_filtered_features.to_numpy()
        np.savez("data/%s_manually_filtered_feature_vectors.npz" % data_type,
                 manually_filtered_features, labels.to_numpy(), filtered_feature_columns)


def create_dataset(z_norm, diff_format, data_path, data_type, feature_extraction, feature_list):
    """
    Iterates through input data and generates an accumulated test / train / validation data set (.npz).

    :param z_norm: whether each sample should be z-normalized
    :param diff_format: whether the samples are provided in a differing format (see above)
    :param data_path: path to sample data
    :param data_type: train | test | validation
    :param feature_extraction: whether feature extraction should be performed
    :param feature_list: list of features to be considered (in case of feature extraction)
    """
    if not feature_extraction:
        create_processed_time_series_dataset(data_path, diff_format, z_norm, data_type)
    else:
        create_feature_vector_dataset(data_path, data_type, feature_list)


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


def file_path(path):
    """
    Returns path if it's valid, raises error otherwise.

    :param path: path to be checked
    :return: feasible path or error
    """
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")


if __name__ == '__main__':
    # input: raw oscilloscope data (one file per recording)
    # output: preprocessed data (one training / testing / validation data file containing data of all recordings)
    parser = argparse.ArgumentParser(description='Preprocess time series data..')
    parser.add_argument('--znorm', action='store_true', help='z-normalize time series')
    parser.add_argument('--diff_format', action='store_true', help='using the "only voltage" format')
    parser.add_argument('--path', type=dir_path, required=True, help='path to the data to be processed')
    parser.add_argument('--feature_extraction', action='store_true', help='apply feature extraction (and selection)')
    parser.add_argument(
        '--feature_list', type=file_path, help='path to the csv file containing the list of features to be considered')
    parser.add_argument(
        '--type', action='store', type=str, help='type of data: ["training", "validation", "test"]', required=True)

    args = parser.parse_args()
    list_of_features = None
    if args.feature_list is not None:
        list_of_features = pd.read_csv(args.feature_list).columns.to_numpy()
    create_dataset(args.znorm, args.diff_format, args.path, args.type, args.feature_extraction, list_of_features)
