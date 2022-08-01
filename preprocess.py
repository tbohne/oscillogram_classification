#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np


def read_oscilloscope_recording(rec_file):
    print("reading oscilloscope recording from", rec_file)

    # label: pos (1) / neg (0)
    label = 1 if "POS" in str(rec_file) else 0

    df = pd.read_csv(rec_file, delimiter=';', na_values=['-∞', '∞'])
    df = df[1:].apply(lambda x: x.str.replace(',', '.')).astype(float).dropna()
    curr_voltages = list(df['Kanal A'].values)

    return label, curr_voltages


def equalize_sample_sizes(voltages):
    min_size = min([len(sample) for sample in voltages])
    # reduce all samples with too many data points
    for i in range(len(voltages)):
        if len(voltages[i]) > min_size:
            remove = len(voltages[i]) - min_size
            voltages[i] = voltages[i][: len(voltages[i]) - remove]


def iterate_through_input_data():
    labels = []
    voltages = []
    for path in Path("data/").glob('**/*.csv'):
        label, curr_voltages = read_oscilloscope_recording(path)
        labels.append(label)
        voltages.append(curr_voltages)
    equalize_sample_sizes(voltages)
    np.savez("data/training_data.npz", np.array(voltages, dtype=object), np.array(labels))


if __name__ == '__main__':
    # input: raw oscilloscope data (one file per recording)
    # output: preprocessed data (one training data file containing data of all recordings)
    iterate_through_input_data()
