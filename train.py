#!/usr/bin/env python
# -*- coding: utf-8 -*-

from training_data import TrainingData
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

EPOCHS = 20
BATCH_SIZE = 32


def visualize_one_example_per_class(x, y):
    classes = np.unique(y, axis=0)
    plt.figure()
    for c in classes:
        x_c = x[y == c]
        plt.plot(x_c[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()


def create_model(input_shape, num_classes):
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


def train_model(x_train, y_train):
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
        verbose=1,
    )
    plot_training_and_validation_loss(history)


def plot_training_and_validation_loss(history):
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
    trained_model = keras.models.load_model("best_model.h5")
    test_loss, test_acc = trained_model.evaluate(x_test, y_test)
    print("test accuracy:", test_acc)
    print("test loss:", test_loss)


if __name__ == '__main__':
    data = TrainingData(np.load("data/training_data.npz", allow_pickle=True))
    print("number of recordings (oscillograms):", len(data))

    x_train = data[:][0]
    y_train = data[:][1]
    visualize_one_example_per_class(x_train, y_train)

    # generally applicable to multivariate time series
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # shuffle training set
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    x_train = np.asarray(x_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')

    model = create_model(x_train.shape[1:], len(np.unique(y_train)))
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    train_model(x_train, y_train)

    # TODO: setup test data
    x_test = x_train
    y_test = y_train
    evaluate_model_on_test_data(x_test, y_test)
