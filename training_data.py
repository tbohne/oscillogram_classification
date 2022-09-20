#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

class TrainingData:
    """
    Training data representation:
        - X: values (time series) or extracted features
        - Y: labels
        - Z: names of the extracted features (optional) - same for all samples
    """

    def __init__(self, data):
        self.X = data['arr_0']
        self.Y = data['arr_1']
        self.Z = None
        if 'arr_2' in data:
            self.Z = data['arr_2']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.Z is not None:
            return self.X[idx], self.Y[idx], self.Z[idx]
        else:
            return self.X[idx], self.Y[idx]
