#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Feb 15, 2017.

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    # This will download the dataset if it does not already exist.
    input_data.read_data_sets('MNIST_data', one_hot=True)
