#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from oscillogram_classification import preprocess


def retrieve_last_conv_layer(trained_model):
    """
    Retrieves the name of the last convolutional layer in the CNN.

    :param trained_model: trained model that produces the prediction to be understood
    :return: name of last conv layer
    """
    for layer in trained_model.layers[::-1]:
        if "conv" in layer.name:
            return layer.name


def compute_gradients_and_last_conv_layer_output(input_array, trained_model, pred_idx):
    """
    Computes the gradients of the predicted class for the input series with respect to the activations of the last conv
    layer and returns them together with the output of the last conv layer.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred_idx: index of the prediction to be analyzed (default is the "best guess")
    :return: (computed gradients, output of last conv layer)
    """
    # remove the last layer's softmax
    trained_model.layers[-1].activation = None

    # model that maps the input to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [trained_model.inputs],
        [trained_model.get_layer(retrieve_last_conv_layer(trained_model)).output, trained_model.output]
    )

    # gradient tape -> API to inspect gradients in tensorflow
    with tf.GradientTape() as tape:
        # compute gradient of the predicted class for the input series with respect to the
        # activations of the last conv layer
        last_conv_layer_output, predictions = grad_model(input_array)
        # no index specified -> use the one with the highest "probability" (the best guess)
        if pred_idx is None:
            pred_idx = tf.argmax(predictions[0])
        pred_value = predictions[:, pred_idx]

    # gradient of the output neuron (top predicted) w.r.t. the output feature map of the last conv layer
    grads = tape.gradient(pred_value, last_conv_layer_output)

    return grads, last_conv_layer_output[0]


def generate_gradcam(input_array, trained_model, pred_idx=None):
    """
    Generates the Grad-CAM (Gradient-weighted Class Activation Map) for the specified input, trained model
    and optionally prediction. It is essentially used to get a sense of what regions of the input the CNN is looking
    at in order to make a prediction.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred_idx: index of the prediction to be analyzed (default is the "best guess")
    :return: class activation map (heatmap) that highlights the most relevant parts for the classification
    """
    grads, last_conv_layer_output = compute_gradients_and_last_conv_layer_output(input_array, trained_model, pred_idx)

    # vector where each entry is the mean intensity of the gradient over a specific feature map channel
    # -> average of gradient values as weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    print("conv out:", last_conv_layer_output.shape)
    print("pooled grads:", pooled_grads[:, tf.newaxis].shape)

    # now a channel could have a high gradient but still a low activation (we want to consider both)
    # thus:
    # multiply each channel in the feature map by how important it is w.r.t. the top
    # predicted class, then sum all the channels to obtain the heatmap class activation
    # [new axis necessary so that the dimensions fit for matrix multiplication]
    cam = last_conv_layer_output @ pooled_grads[:, tf.newaxis]
    print("cam shape:", cam.shape)

    # get back to time series shape (1D) -> remove dimension of size 1
    cam = tf.squeeze(cam)
    # for visualization purpose, normalize heatmap
    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam.numpy()


def generate_hirescam(input_array, trained_model, pred_idx=None):
    """
    Generates the HiResCAM for the specified input, trained model and optionally prediction. It is essentially used to
    get a sense of what regions of the input the CNN is looking at in order to make a prediction.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred_idx: index of the prediction to be analyzed (default is the "best guess")
    :return: class activation map (heatmap) that highlights the most relevant parts for the classification
    """
    grads, last_conv_layer_output = compute_gradients_and_last_conv_layer_output(input_array, trained_model, pred_idx)
    grads = tf.squeeze(grads)

    # element-wise product between the raw gradients and feature maps
    cam = np.multiply(last_conv_layer_output, grads)
    # sum over feature dimensions
    cam = cam.sum(axis=1)

    # for visualization purpose, normalize heatmap
    cam = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam.numpy()


def plot_cam(cam, voltage_vals):
    """
    Visualizes the class activation map (heatmap).

    :param cam: class activation map to be visualized
    :param voltage_vals: voltage values to be visualized
    """
    plt.rcParams["figure.figsize"] = 10, 4
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)

    # bounding box in data coordinates that the image will fill (left, right, bottom, top)
    extent = [0, cam.shape[0], 0, 1]

    ax.imshow(cam[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    data_points = [i for i in range(len(voltage_vals))]
    ax2.plot(data_points, voltage_vals)
    plt.tight_layout()
    plt.show()


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
    parser = argparse.ArgumentParser(description='Apply diff. CAM methods to trained model to understand predictions..')
    parser.add_argument('--altering_format', action='store_true', help='using the "only voltage" format')
    parser.add_argument('--sample_path', type=file_path, required=True)
    parser.add_argument('--znorm', action='store_true', help='z-normalize time series')
    parser.add_argument('--model_path', type=file_path, required=True)
    parser.add_argument('--method', action='store', type=str, help='method: ["gradcam", "hirescam"]', required=True)

    args = parser.parse_args()

    if args.altering_format:
        _, voltages = preprocess.read_voltage_only_format_recording(args.sample_path)
    else:
        _, voltages = preprocess.read_oscilloscope_recording(args.sample_path)

    if args.znorm:
        voltages = preprocess.z_normalize_time_series(voltages)

    model = keras.models.load_model(args.model_path)
    net_input_size = model.layers[0].output_shape[0][1]

    assert net_input_size == len(voltages)
    # TODO: think about whether it makes sense to fix input size
    # if len(voltages) > net_input_size:
    #     remove = len(voltages) - net_input_size
    #     voltages = voltages[: len(voltages) - remove]

    net_input = np.asarray(voltages).astype('float32')
    net_input = net_input.reshape((net_input.shape[0], 1))

    print("input shape:", net_input.shape)
    print(model.summary())

    # EXPLAIN PREDICTION WITH GRAD-CAM
    prediction = model.predict(np.array([net_input]))
    print("prediction:", prediction)

    heatmap = None
    if args.method == "gradcam":
        heatmap = generate_gradcam(np.array([net_input]), model)
    elif args.method == "hirescam":
        heatmap = generate_hirescam(np.array([net_input]), model)
    else:
        print("specified unknown CAM method:", args.method)

    if heatmap is not None:
        plot_cam(heatmap, voltages)
