#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import preprocess


def generate_gradcam_heatmap(input_array, trained_model, last_conv_layer, pred_idx=None):
    # model that maps the input to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [trained_model.inputs], [trained_model.get_layer(last_conv_layer).output, trained_model.output]
    )

    # compute gradient of the predicted class for input series with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(input_array)
        if pred_idx is None:
            pred_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_idx]

    # gradient of the output neuron (top predicted) w.r.t. the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # multiply each channel in the feature map array by "how important this channel is" with regard to the top
    # predicted class, then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # for visualization purpose, normalize heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def plot_heatmap(heatmap):
    plt.rcParams["figure.figsize"] = 10, 4
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)

    # bounding box in data coordinates that the image will fill (left, right, bottom, top)
    extent = [0, heatmap.shape[0], 0, 1]

    ax.imshow(heatmap[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    data_points = [i for i in range(len(voltages))]
    ax2.plot(data_points, voltages)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    _, voltages = preprocess.read_oscilloscope_recording("data/NEG_WVWZZZAUZHP535532.csv")
    voltages = preprocess.z_normalize_time_series(voltages)

    model = keras.models.load_model("best_model.h5")

    # fix input size
    net_input_size = model.layers[0].output_shape[0][1]
    if len(voltages) > net_input_size:
        remove = len(voltages) - net_input_size
        voltages = voltages[: len(voltages) - remove]

    net_input = np.asarray(voltages).astype('float32')
    net_input = net_input.reshape((net_input.shape[0], 1))

    print("input shape:", net_input.shape)

    print(model.summary())
    last_conv_layer_name = "conv1d_2"

    # EXPLAINABILITY

    # remove last layer's softmax
    model.layers[-1].activation = None

    prediction = model.predict(np.array([net_input]))
    print("Predicted:", prediction)

    # generate class activation heatmap
    heatmap = generate_gradcam_heatmap(np.array([net_input]), model, last_conv_layer_name)

    print("heatmap shape:", heatmap.shape)

    plot_heatmap(heatmap)
