#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

hyperparameter_config = {
    "batch_size": 4,
    "learning_rate": 0.001,
    "optimizer": "keras.optimizers.Adam",
    "epochs": 100,
    "model": "FCN_binary",
    "loss_function": "binary_crossentropy",  # sparse_categorical_crossentropy
    "accuracy_metric": "binary_accuracy"  # sparse_categorical_accuracy
}
