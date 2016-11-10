#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Oct 25, 2016.

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

SPLITS_TO_SIZES = {
    "train": 3320,
    "validation": 350,
}

def write_label_file(class_names, filename):
    with open(filename, 'w') as fp:
        for label, class_name in enumerate(class_names):
            fp.write("%d:%s\n" % (label, class_name))


def read_label_file(filename):
    labels_to_names = {}
    with open(filename, 'r') as fp:
        for line in fp:
            parts = line.split(':')
            labels_to_names[int(parts[0])] = parts[1].strip()
    return labels_to_names


def get_dataset(split_name, dataset_dir):
    assert split_name in ["train", "validation"]
    file_pattern = os.path.join(dataset_dir, "flowers_%s_*.tfrecord" % split_name)
    reader = tf.TFRecordReader
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/format": tf.FixedLenFeature((), tf.string, default_value="png"),
        "image/class/label": tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    items_to_handlers = {
        "image": slim.tfexample_decoder.Image(
            image_key = "image/encoded",
            format_key = "image/format"),
        "label": slim.tfexample_decoder.Tensor("image/class/label"),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = read_label_file(os.path.join(dataset_dir, "labels.txt"))
    return slim.dataset.Dataset(
        data_sources = file_pattern,
        reader = reader,
        decoder = decoder,
        num_samples = SPLITS_TO_SIZES[split_name],
        items_to_descriptions = {"image": "A color image of varying size.",
                                 "label": "A single integer between 0 and 4"},
        num_classes = 5,
        labels_to_names = labels_to_names)
