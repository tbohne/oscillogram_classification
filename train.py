#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from wandb.keras import WandbCallback

import models
from config import api_key, run_config
from training_data import TrainingData


def set_up_wandb(wandb_config):
    """
    Setup for 'weights and biases'.

    :param wandb_config: configuration to be used
    """
    wandb.login(key=api_key.wandb_api_key)
    wandb.init(project="Oscillogram Classification", config=wandb_config)


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


def train_keras_model(model, x_train, y_train, x_val, y_val):
    """
    Trains the specified Keras model on the specified data.

    :param model: model to be trained
    :param x_train: training data samples
    :param y_train: training data labels
    :param x_val: validation data samples
    :param y_val: validation data labels
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        WandbCallback()
    ]

    optimizer = eval(wandb.config["optimizer"])(learning_rate=wandb.config["learning_rate"])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=wandb.config["batch_size"],
        epochs=wandb.config.epochs,
        callbacks=callbacks,
        validation_split=0.2,
        validation_data=(x_val, y_val),
        verbose=1,
    )
    plot_training_and_validation_loss(history)


def train_model(model, x_train, y_train, x_val, y_val):
    """
    Trains the selected model (classifier).

    :param model: model to be trained
    :param x_train: training samples
    :param y_train: corresponding training labels
    :param x_val: validation samples
    :param y_val: corresponding validation labels
    """
    print("training model:")
    print("total training samples:", len(x_train))
    for c in np.unique(y_train, axis=0):
        assert len(x_train[y_train == c]) > 0
        print("training samples for class", str(c), ":", len(x_train[y_train == c]))
    print("total validation samples:", len(x_val))
    for c in np.unique(y_val, axis=0):
        assert len(x_val[y_val == c]) > 0
        print("validation samples for class", str(c), ":", len(x_val[y_val == c]))

    x_train = np.squeeze(x_train)
    x_val = np.squeeze(x_val)

    print("train shape:", x_train.shape)
    print("val shape:", x_val.shape)

    if x_train.shape[1] > x_val.shape[1]:
        # pad validation feature vector
        x_val = np.pad(x_val, ((0, 0), (0, x_train.shape[1] - x_val.shape[1])), mode='constant')
    elif x_val.shape[1] > x_train.shape[1]:
        # pad training feature vector
        x_train = np.pad(x_train, ((0, 0), (0, x_val.shape[1] - x_train.shape[1])), mode='constant')

    print("train shape:", x_train.shape)
    print("val shape:", x_val.shape)

    assert x_train.shape[1] == x_val.shape[1]

    if 'keras' in str(type(model)):
        train_keras_model(model, x_train, y_train, x_val, y_val)
    else:
        model.fit(x_train, y_train)


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


def evaluate_model_on_test_data(x_test, y_test, model):
    """
    Evaluates the trained model on the specified test data.

    :param x_test: test samples
    :param y_test: corresponding labels
    :param model: trained model to be evaluated
    """
    print("evaluating model:")
    print("total test samples:", len(x_test))
    for c in np.unique(y_test, axis=0):
        assert len(x_test[y_test == c]) > 0
        print("test samples for class", str(c), ":", len(x_test[y_test == c]))

    print("x_test shape:", x_test.shape)

    if 'keras' in str(type(model)):
        # should be the same, but read from file
        model = keras.models.load_model("best_model.h5")
        expected_feature_vector_len = model.layers[0].output_shape[0][1]
    else:
        expected_feature_vector_len = model.n_features_in_

    # feature vectors not necessarily of same length
    if x_test.shape[1] < expected_feature_vector_len:
        # pad feature vector
        x_test = np.pad(x_test, ((0, 0), (0, expected_feature_vector_len - x_test.shape[1])), mode='constant')
    # test samples should match model input length
    assert x_test.shape[1] == expected_feature_vector_len

    if 'keras' in str(type(model)):
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("test accuracy:", test_acc)
        print("test loss:", test_loss)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    else:
        y_pred_test = model.predict(x_test)
        print("test accuracy:", accuracy_score(y_test, y_pred_test))
        print("CLASSIFICATION REPORT:")
        res_dict = classification_report(y_test, y_pred_test, output_dict=True)
        for k in res_dict.keys():
            print("k:", k)
            print(res_dict[k])


def perform_consistency_check(train_data, val_data, test_data, z_train, z_val, z_test):
    """
    Performs a consistency check for the provided data:
        - all three (train, val, test) should either provide feature info or not
        - if the data consists of feature vectors, all three sets should contain
          exactly the same features in the same order

    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: test dataset
    :param z_train: features of the training dataset (or empty)
    :param z_val: features of the validation dataset (or empty)
    :param z_test: features of the test dataset (or empty)
    """
    print("performing consistency check..")
    # equal number of dimensions (either all have feature info or none)
    assert len(train_data[:]) == len(val_data[:]) == len(test_data[:])
    # equal number of considered features
    assert len(z_train) == len(z_val) == len(z_test)
    # check whether all features are the same (+ same order)
    for i in range(len(z_train)):
        assert z_train[i] == z_val[i] == z_test[i]
    print("consistency check passed..")


def prepare_data(train_data_path, val_data_path, test_data_path, keras_model):
    """
    Prepares the data for the training / evaluation process.

    :param train_data_path: path to read training data from
    :param val_data_path: path to read validation data from
    :param test_data_path: path to read test data from
    :param keras_model: whether the data is prepared for a keras model
    :return: (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    data = TrainingData(np.load(train_data_path, allow_pickle=True))
    x_train = data[:][0]
    y_train = data[:][1]
    if len(data[:]) == 3:
        z_train = data[:][2]

    visualize_n_samples_per_class(x_train, y_train)

    if keras_model:
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
    if len(val_data[:]) == 3:
        z_val = data[:][2]

    # read test data
    test_data = TrainingData(np.load(test_data_path, allow_pickle=True))
    x_test = test_data[:][0]
    y_test = test_data[:][1]
    if len(test_data[:]) == 3:
        z_test = test_data[:][2]

    perform_consistency_check(data, val_data, test_data, z_train, z_val, z_test)

    return x_train, y_train, x_val.astype('float32'), y_val.astype('float32'), \
        x_test.astype('float32'), y_test.astype('float32')


def train_procedure(train_path, val_path, test_path, hyperparameter_config=run_config.hyperparameter_config):
    """
    Initiates the training and evaluation procedures.

    :param train_path: path to training data
    :param val_path: path to validation data
    :param test_path: path to test data
    :param hyperparameter_config: hyperparameter specification
    """
    set_up_wandb(hyperparameter_config)
    keras_model = wandb.config["model"] in ["FCN", "ResNet"]
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(train_path, val_path, test_path, keras_model)
    model = models.create_model(x_train.shape[1:], len(np.unique(y_train)), architecture=wandb.config["model"])

    if 'keras' in str(type(model)):
        keras.utils.plot_model(model, to_file="img/model.png", show_shapes=True)

    train_model(model, x_train, y_train, x_val, y_val)
    evaluate_model_on_test_data(x_test, y_test, model)


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
    parser = argparse.ArgumentParser(description='Training model with time series data..')
    parser.add_argument('--train_path', type=file_path, required=True)
    parser.add_argument('--val_path', type=file_path, required=True)
    parser.add_argument('--test_path', type=file_path, required=True)
    args = parser.parse_args()

    train_procedure(args.train_path, args.val_path, args.test_path)
