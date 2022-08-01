#!/usr/bin/env python
# -*- coding: utf-8 -*-

from training_data import TrainingData
import numpy as np
import matplotlib.pyplot as plt


def visualize_one_example_per_class(x, y):
    classes = np.unique(y, axis=0)
    plt.figure()
    for c in classes:
        x_c = x[y == c]
        plt.plot(x_c[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()


if __name__ == '__main__':
    data = TrainingData(np.load("data/training_data.npz", allow_pickle=True))
    print("number of recordings (oscillograms):", len(data))

    x_train = data[:][0]
    y_train = data[:][1]
    visualize_one_example_per_class(x_train, y_train)
