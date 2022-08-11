#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_oscilloscope_recording(rec_file):
    """
    Reads the oscilloscope recording from the specified file in standard format.

    :param rec_file: oscilloscope recording file
    :return: label, list of voltage values (time series)
    """
    print("reading oscilloscope recording from", rec_file)
    # label: pos (1) / neg (0)
    label = 1 if "POS" in str(rec_file) else 0
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
    # label: pos (1) / neg (0)
    label = 1 if "POS" in str(rec_file) else 0
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


def iterate_through_input_data(z_norm, altering_format, data_path, data_type):
    """
    Iterates through input data and generates an accumulated test / train / validation data set (.npz).

    :param z_norm: whether each sample should be z-normalized
    :param altering_format: whether the samples are provided in altering format (see above)
    :param data_path: path to sample data
    :param data_type: train | test | validation
    """
    labels = []
    voltage_series = []
    for path in Path(data_path).glob('**/*.csv'):
        label, curr_voltages = read_oscilloscope_recording(
            path) if not altering_format else read_voltage_only_format_recording(path)
        labels.append(label)
        if z_norm:
            curr_voltages = z_normalize_time_series(curr_voltages)
        voltage_series.append(curr_voltages)
    equalize_sample_sizes(voltage_series)
    np.savez("data/%s_data.npz" % data_type, np.array(voltage_series, dtype=object), np.array(labels))


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


if __name__ == '__main__':
    # input: raw oscilloscope data (one file per recording)
    # output: preprocessed data (one training / testing / validation data file containing data of all recordings)
    parser = argparse.ArgumentParser(description='Preprocess time series data..')
    parser.add_argument('--znorm', action='store_true', help='z-normalize time series')
    parser.add_argument('--altering_format', action='store_true', help='using the "only voltage" format')
    parser.add_argument('--path', type=dir_path, required=True)
    parser.add_argument(
        '--type', action='store', type=str, help='type of data: ["training", "validation", "test"]', required=True)

    args = parser.parse_args()
    iterate_through_input_data(args.znorm, args.altering_format, args.path, args.type)
