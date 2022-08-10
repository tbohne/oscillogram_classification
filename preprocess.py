#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_oscilloscope_recording(rec_file):
    print("reading oscilloscope recording from", rec_file)
    # label: pos (1) / neg (0)
    label = 1 if "POS" in str(rec_file) else 0
    df = pd.read_csv(rec_file, delimiter=';', na_values=['-∞', '∞'])
    df = df[1:].apply(lambda x: x.str.replace(',', '.')).astype(float).dropna()
    curr_voltages = list(df['Kanal A'].values)
    return label, curr_voltages


def read_voltage_only_format_recording(rec_file):
    print("reading oscilloscope recording from", rec_file)
    # label: pos (1) / neg (0)
    label = 1 if "POS" in str(rec_file) else 0
    a = pd.read_csv(rec_file, delimiter=",", na_values=["-∞", "∞"], names=["Kanal A"])
    curr_voltages = list(a['Kanal A'].values)
    return label, curr_voltages


def equalize_sample_sizes(voltages):
    min_size = min([len(sample) for sample in voltages])
    # reduce all samples with too many data points
    for i in range(len(voltages)):
        if len(voltages[i]) > min_size:
            remove = len(voltages[i]) - min_size
            voltages[i] = voltages[i][: len(voltages[i]) - remove]


def z_normalize_time_series(series):
    return (series - np.mean(series)) / np.std(series)


def iterate_through_input_data(z_norm, altering_format, data_path, data_type):
    labels = []
    voltages = []
    for path in Path(data_path).glob('**/*.csv'):
        label, curr_voltages = read_oscilloscope_recording(path) if not altering_format else read_voltage_only_format_recording(path)
        labels.append(label)
        if z_norm:
            curr_voltages = z_normalize_time_series(curr_voltages)
        voltages.append(curr_voltages)
    equalize_sample_sizes(voltages)
    np.savez("data/%s_data.npz" % data_type, data_type, np.array(voltages, dtype=object), np.array(labels))


def dir_path(path):
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
