#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

class TrainingData:
    """
    Training data representation:
        - X: data points (time series)
        - Y: labels
    """

    def __init__(self, data):
        self.X = data['arr_0']
        self.Y = data['arr_1']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
