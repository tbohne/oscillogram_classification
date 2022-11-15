#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from oscillogram_classification import preprocess
from tensorflow import keras
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore

METHODS = ["gradcam", "hirescam", "tf-keras-gradcam", "tf-keras-gradcam++", "tf-keras-scorecam",
           "tf-keras-smoothgrad", "tf-keras-layercam", "all"]


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


def normalize_heatmap(cam):
    """
    Normalizes the heatmap for visualization purposes.

    :param cam: class activation map (heatmap)
    :return: normalized CAM
    """
    if len(np.unique(cam)) > 1 or np.unique(cam)[0] != 0:
        return tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return cam


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

    # print("conv out:", last_conv_layer_output.shape)
    # print("pooled grads:", pooled_grads[:, tf.newaxis].shape)

    # now a channel could have a high gradient but still a low activation (we want to consider both)
    # thus:
    # multiply each channel in the feature map by how important it is w.r.t. the top
    # predicted class, then sum all the channels to obtain the heatmap class activation
    # [new axis necessary so that the dimensions fit for matrix multiplication]
    cam = last_conv_layer_output @ pooled_grads[:, tf.newaxis]

    # get back to time series shape (1D) -> remove dimension of size 1
    cam = tf.squeeze(cam)
    cam = normalize_heatmap(cam)
    return cam.numpy()


def tf_keras_gradcam(input_array, trained_model, pred):
    """
    Generates Grad-CAM heatmap using the tf-keras-vis library.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred: considered prediction
    :return: class activation map (heatmap) that highlights the most relevant parts for the classification
    """
    num_classes = len(pred[0])
    print("#classes:", num_classes)
    print("prediction:", pred[0][0])

    gradcam = Gradcam(trained_model, model_modifier=ReplaceToLinear(), clone=True)
    if num_classes > 1:
        score = CategoricalScore([np.argmax(pred)])
    else:
        score = BinaryScore(pred[0][0] > 0.5)
    cam = gradcam(score, input_array, penultimate_layer=-1)
    cam = tf.squeeze(cam)
    cam = normalize_heatmap(cam)
    return cam.numpy()


def tf_keras_gradcam_plus_plus(input_array, trained_model, pred):
    """
    Generates Grad-CAM++ heatmap using the tf-keras-vis library.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred: considered prediction
    :return: class activation map (heatmap) that highlights the most relevant parts for the classification
    """
    num_classes = len(pred[0])
    print("#classes:", num_classes)
    print("prediction:", pred[0][0])

    gradcam = GradcamPlusPlus(trained_model, model_modifier=ReplaceToLinear(), clone=True)
    if num_classes > 1:
        score = CategoricalScore([np.argmax(pred)])
    else:
        score = BinaryScore(pred[0][0] > 0.5)
    cam = gradcam(score, input_array, penultimate_layer=-1)
    cam = tf.squeeze(cam)
    cam = normalize_heatmap(cam)
    return cam.numpy()


def tf_keras_scorecam(input_array, trained_model, pred):
    """
    Generates ScoreCAM heatmap using the tf-keras-vis library.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred: considered prediction
    :return: class activation map (heatmap) that highlights the most relevant parts for the classification
    """
    num_classes = len(pred[0])
    print("#classes:", num_classes)
    print("prediction:", pred[0][0])

    scorecam = Scorecam(trained_model, model_modifier=ReplaceToLinear())
    if num_classes > 1:
        # idx of the class to be considered
        score = CategoricalScore([np.argmax(pred)])
    else:
        score = BinaryScore(pred[0][0] > 0.5)
    cam = scorecam(score, input_array, penultimate_layer=-1)
    cam = tf.squeeze(cam)
    cam = normalize_heatmap(cam)
    return cam.numpy()


def tf_keras_layercam(input_array, trained_model, pred):
    """
    Generates LayerCAM heatmap using the tf-keras-vis library.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred: considered prediction
    :return: class activation map (heatmap) that highlights the most relevant parts for the classification
    """
    num_classes = len(pred[0])
    print("#classes:", num_classes)
    print("prediction:", pred[0][0])

    layercam = Layercam(trained_model)
    if num_classes > 1:
        score = CategoricalScore([np.argmax(pred)])
    else:
        score = BinaryScore(pred[0][0] > 0.5)
    cam = layercam(score, input_array, penultimate_layer=-1)
    cam = tf.squeeze(cam)
    cam = normalize_heatmap(cam)
    return cam.numpy()


def tf_keras_smooth_grad(input_array, trained_model, pred):
    """
    Generates SmoothGrad saliency map using the tf-keras-vis library.

    :param input_array: input to understand prediction for
    :param trained_model: trained model that produces the prediction to be understood
    :param pred: considered prediction
    :return: saliency map that highlights the most relevant parts for the classification
    """
    num_classes = len(pred[0])
    print("#classes:", num_classes)
    print("prediction:", pred[0][0])

    saliency = Saliency(trained_model, model_modifier=ReplaceToLinear(), clone=True)

    if num_classes > 1:
        score = CategoricalScore([np.argmax(pred)])
    else:
        score = BinaryScore(pred[0][0] > 0.5)
    saliency_map = saliency(score, input_array, smooth_samples=20, smooth_noise=0.20)
    cam = tf.squeeze(saliency_map)
    cam = normalize_heatmap(cam)
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
    cam = normalize_heatmap(cam)
    return cam.numpy()


def plot_heatmaps_as_overlay(cams, voltage_vals, title):
    """
    Visualizes the class activation maps (heatmaps) - time series as overlay.

    :param cams: dictionary containing the class activation maps to be visualized (+ method names)
    :param voltage_vals: voltage values to be visualized
    :param title: window title, e.g., recorded vehicle component
    """
    plt.rcParams["figure.figsize"] = len(cams) * 10, 3
    fig, axes = plt.subplots(nrows=1, ncols=len(cams), sharex=True, sharey=True)
    fig.canvas.set_window_title(title)
    # bounding box in data coordinates that the image will fill (left, right, bottom, top)
    extent = [0, cams[list(cams.keys())[0]].shape[0], np.floor(np.min(voltage_vals)), np.ceil(np.max(voltage_vals))]
    data_points = np.array([i for i in range(len(voltage_vals))])

    if len(cams) == 1:
        axes = [axes]

    for i in range(len(cams)):
        axes[i].set_xlim(extent[0], extent[1])
        axes[i].title.set_text(list(cams.keys())[i])
        # heatmap
        axes[i].imshow(cams[list(cams.keys())[i]][np.newaxis, :], cmap="plasma", aspect="auto", alpha=.75,
                       extent=extent)
        # data
        axes[i].plot(data_points, voltage_vals, '#000000')

    plt.tight_layout()
    plt.show()


def plot_heatmaps(cams, voltage_vals, title):
    """
    Visualizes the class activation maps (heatmaps).

    :param cams: dictionary containing the class activation maps to be visualized (+ method names)
    :param voltage_vals: voltage values to be visualized
    :param title: window title, e.g., recorded vehicle component
    """
    plt.rcParams["figure.figsize"] = len(cams) * 10, 4
    fig, axes = plt.subplots(nrows=2, ncols=len(cams), sharex=True, sharey=True)
    fig.canvas.set_window_title(title)

    # bounding box in data coordinates that the image will fill (left, right, bottom, top)
    extent = [0, cams[list(cams.keys())[0]].shape[0], np.floor(np.min(voltage_vals)), np.ceil(np.max(voltage_vals))]
    data_points = [i for i in range(len(voltage_vals))]

    for i in range(len(cams)):
        # first row (heatmaps)
        curr_above = axes[0][i] if len(cams) > 1 else axes[0]
        # curr_above.set_yticks([])
        curr_above.set_xlim(extent[0], extent[1])
        curr_above.title.set_text(list(cams.keys())[i])

        # second row (voltages)
        curr_below = axes[1][i] if len(cams) > 1 else axes[1]

        curr_above.imshow(cams[list(cams.keys())[i]][np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
        curr_below.plot(data_points, voltage_vals, '#000000')

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
    parser.add_argument('--diff_format', action='store_true', help='using the "only voltage" format')
    parser.add_argument('--sample_path', type=file_path, required=True)
    parser.add_argument('--znorm', action='store_true', help='z-normalize time series')
    parser.add_argument('--model_path', type=file_path, required=True)
    parser.add_argument('--method', action='store', type=str, help='methods: %s' % METHODS, required=True)
    parser.add_argument('--overlay', action='store_true', help='overlay heatmap and time series')

    args = parser.parse_args()

    if args.diff_format:
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
    # elif net_input_size > len(voltages):
    #     for _ in range(net_input_size - len(voltages)):
    #         voltages.append(0)

    net_input = np.asarray(voltages).astype('float32')
    net_input = net_input.reshape((net_input.shape[0], 1))

    print("input shape:", net_input.shape)
    print(model.summary())

    # EXPLAIN PREDICTION WITH GRAD-CAM
    prediction = model.predict(np.array([net_input]))
    print("PREDICTION:", prediction)
    print("predicted class", np.argmax(prediction), " with score", np.max(prediction))

    heatmaps = {}
    if args.method == "gradcam":
        heatmaps[args.method] = generate_gradcam(np.array([net_input]), model)
    elif args.method == "tf-keras-gradcam":
        heatmaps[args.method] = tf_keras_gradcam(np.array([net_input]), model, prediction)
    elif args.method == "tf-keras-gradcam++":
        heatmaps[args.method] = tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction)
    elif args.method == "tf-keras-scorecam":
        heatmaps[args.method] = tf_keras_scorecam(np.array([net_input]), model, prediction)
    elif args.method == "tf-keras-layercam":
        heatmaps[args.method] = tf_keras_layercam(np.array([net_input]), model, prediction)
    elif args.method == "tf-keras-smoothgrad":
        heatmaps[args.method] = tf_keras_smooth_grad(np.array([net_input]), model, prediction)
    elif args.method == "hirescam":
        heatmaps[args.method] = generate_hirescam(np.array([net_input]), model)
    elif args.method == "all":
        # not needed as it returns exactly the same heatmap as tf-keras-gradcam
        # heatmaps["gradcam"] = generate_gradcam(np.array([net_input]), model)
        heatmaps["tf-keras-gradcam"] = tf_keras_gradcam(np.array([net_input]), model, prediction)
        heatmaps["tf-keras-gradcam++"] = tf_keras_gradcam_plus_plus(np.array([net_input]), model, prediction)
        # TODO: deactivated for now (can't yet deal with one-neuron output binary classification)
        # heatmaps["hirescam"] = generate_hirescam(np.array([net_input]), model)
        heatmaps["tf-keras-scorecam"] = tf_keras_scorecam(np.array([net_input]), model, prediction)
        heatmaps["tf-keras-layercam"] = tf_keras_layercam(np.array([net_input]), model, prediction)
        # TODO: should be checked later on (no reasonable heatmaps)
        # heatmaps["tf-keras-smoothgrad"] = tf_keras_smooth_grad(np.array([net_input]), model, prediction)
    else:
        print("specified unknown CAM method:", args.method)

    if len(heatmaps) > 0:
        if args.overlay:
            plot_heatmaps_as_overlay(heatmaps, voltages, 'test_plot')
        else:
            plot_heatmaps(heatmaps, voltages, 'test_plot')
