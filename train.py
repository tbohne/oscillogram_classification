#!/usr/bin/env python
# -*- coding: utf-8 -*-

from training_data import TrainingData
import numpy as np


if __name__ == '__main__':
    data = TrainingData(np.load("data/training_data.npz", allow_pickle=True))
    print("number of recordings (oscillograms):", len(data))

    for oscillogram in data:
        data, label = oscillogram
        print(label)
