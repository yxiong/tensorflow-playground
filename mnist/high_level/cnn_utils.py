#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Feb 15, 2017.

import tensorflow as tf


def model_fn(features, labels, mode):
    """The model function fed to Estimator."""
    logits = inference(features, 0.5 if mode==tf.contrib.learn.ModeKeys.TRAIN else 1.0)
    loss = cross_entropy_loss(logits, labels)
    train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = tf.contrib.framework.get_global_step(),
            optimizer = "Adam",
            learning_rate = 1e-4)
    return tf.contrib.learn.ModelFnOps(
            mode = mode,
            predictions = tf.argmax(logits, 1),
            loss = loss,
            train_op = train_op,
            eval_metric_ops = {"accuracy": accuracy(logits, labels)})


def inference(x, keep_prob):
    """Build a CNN for inference.

    Args:
      x: a placeholder node of shape [None, 784].
      keep_prob: a control node for dropout, a placeholder of type float.

    Returns:
      y: output node for the probability of each label, of shape [None, 10]"""
    x_image = tf.reshape(x, [-1,28,28,1])

    # First convolutional layer.
    h_conv1 = tf.layers.conv2d(
            inputs = x_image,
            filters = 32,
            kernel_size = [5,5],
            padding = "SAME",
            activation = tf.nn.relu,
            name = "conv1")
    h_pool1 = tf.layers.max_pooling2d(
            inputs = h_conv1,
            pool_size = [2,2],
            strides = [2,2],
            name = "pool1")

    # Second convolutional layer.
    h_conv2 = tf.layers.conv2d(
            inputs = h_pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = "SAME",
            activation = tf.nn.relu,
            name = "conv2")
    h_pool2 = tf.layers.max_pooling2d(
            inputs = h_conv2,
            pool_size = [2,2],
            strides = [2,2],
            name = "pool2")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="flatten2")

    # Fully connected layer with a dropout.
    h_fc1 = tf.layers.dense(
            inputs = h_pool2_flat,
            units = 1024,
            activation = tf.nn.relu,
            name = "dense1")
    h_fc1_drop = tf.layers.dropout(h_fc1, keep_prob, name="dropout1")

    # Readout layer: fully connected followed by softmax.
    y_fc2 = tf.layers.dense(
            inputs = h_fc1_drop,
            units = 10,
            activation = tf.nn.softmax,
            name = "dense2")

    # Rename the output node.
    y = tf.identity(y_fc2, name="output")

    return y


def cross_entropy_loss(y, y_):
    """Calculate the cross-entropy loss from the inference output and labels.

    Args:
      y: the output probabilities from inference()
      y_: the one-hot representation of labels, of shape [None, 10]

    Returns:
      loss: loss tensor of type float
    """
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1], name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
    return loss


def accuracy(y, y_):
    """Calculate the accuracy from the inference output and labels."""
    cross_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="accuracy")
    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32), name="accuracy_mean")
    return accuracy
