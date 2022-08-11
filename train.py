#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from training_data import TrainingData

EPOCHS = 50
BATCH_SIZE = 32


def visualize_n_samples_per_class(x, y):
    """
    Iteratively visualizes one sample per class until the user enters '+'.

    :param x: sample series
    :param y: corresponding labels
    """
    plt.figure()
    classes = np.unique(y, axis=0)
    samples_by_class = {c: x[y == c] for c in classes}

    for sample in range(len(samples_by_class[classes[0]])):
        key = input("Enter '+' to see another sample per class\n")
        if key != "+":
            break
        for c in classes:
            plt.plot(samples_by_class[c][sample], label="class " + str(c))
        plt.legend(loc="best")
        plt.show()
        plt.close()


def create_model(input_shape, num_classes):
    """
    Defines the CNN architecture to be worked with.

    :param input_shape: shape of the input layer
    :param num_classes: number of unique classes to be considered
    :return: CNN model
    """
    # input shape -> number of data points per sample
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train_model(model, x_train, y_train, x_val, y_val):
    """
    Trains the CNN model.

    :param model: CNN model to be trained
    :param x_train: training samples
    :param y_train: corresponding training labels
    :param x_val: validation samples
    :param y_val: corresponding validation labels
    """
    print("training model:")
    print("total training samples:", len(x_train))
    for c in np.unique(y_train, axis=0):
        print("training samples for class", str(c), ":", len(x_train[y_train == c]))
    print("total validation samples:", len(x_val))
    for c in np.unique(y_val, axis=0):
        print("validation samples for class", str(c), ":", len(x_val[y_val == c]))

    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_split=0.2,
        validation_data=(x_val, y_val),
        verbose=1,
    )
    plot_training_and_validation_loss(history)


def plot_training_and_validation_loss(history):
    """
    Plots the learning curves.

    :param history: training history
    """
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()


def evaluate_model_on_test_data(x_test, y_test):
    """
    Evaluates the trained model on the specified test data.

    :param x_test: test samples
    :param y_test: corresponding labels
    """
    print("evaluating model:")
    print("total test samples:", len(x_test))
    for c in np.unique(y_test, axis=0):
        print("test samples for class", str(c), ":", len(x_test[y_test == c]))

    trained_model = keras.models.load_model("best_model.h5")
    # test samples should match model input length
    assert x_test.shape[1] == trained_model.layers[0].output_shape[0][1]
    # TODO: think about whether it should be able to deal with not matching input lengths
    #       if so: x_test[:, :trained_model.layers[0].output_shape[0][1]]
    test_loss, test_acc = trained_model.evaluate(x_test, y_test)
    print("test accuracy:", test_acc)
    print("test loss:", test_loss)


def prepare_and_train_model(train_data_path, val_data_path):
    """
    Prepares and initiates the training process.

    :param train_data_path: path to read training data from
    :param val_data_path: path to read validation data from
    """
    data = TrainingData(np.load(train_data_path, allow_pickle=True))
    x_train = data[:][0]
    y_train = data[:][1]

    visualize_n_samples_per_class(x_train, y_train)

    # generally applicable to multivariate time series
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # shuffle training set
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    x_train = np.asarray(x_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')

    # read validation data
    val_data = TrainingData(np.load(val_data_path, allow_pickle=True))
    x_val = val_data[:][0]
    y_val = val_data[:][1]

    model = create_model(x_train.shape[1:], len(np.unique(y_train)))
    keras.utils.plot_model(model, to_file="img/model.png", show_shapes=True)
    train_model(model, x_train, y_train, x_val.astype('float32'), y_val.astype('float32'))


def evaluate_model(test_data_path):
    """
    Initiates model evaluation.

    :param test_data_path: path to read test data from
    """
    test_data = TrainingData(np.load(test_data_path, allow_pickle=True))
    x_test = test_data[:][0]
    y_test = test_data[:][1]
    evaluate_model_on_test_data(x_test.astype('float32'), y_test.astype('float32'))


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
    parser = argparse.ArgumentParser(description='Training CNN with time series data..')
    parser.add_argument('--train_path', type=file_path, required=True)
    parser.add_argument('--val_path', type=file_path, required=True)
    parser.add_argument('--test_path', type=file_path, required=True)
    args = parser.parse_args()

    prepare_and_train_model(args.train_path, args.val_path)
    evaluate_model(args.test_path)
