# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from scipy import ndimage
from six.moves import urllib

from hmd_load_data import *
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchvision.transforms as transforms
from torch.autograd import Variable as V

# Params for HDM
NUM_CHANNELS = 1
NUM_ACTION_LABELS = 130



def get_hmd_data(test_id=0, dataset_dir='HDM_data/'):

    x_train, y_train, x_test, y_test = get_hmd(test_id, dataset_dir)
    print("x_train ",x_train.shape, y_train.shape) 

    print("max1 ",x_train.max()) #1
    print("min1 ",x_train.min()) #0

    print("max2 ",x_test.max()) #1
    print("min2 ",x_test.min()) #0


    x_train_1 = x_train.reshape(x_train.shape[0], -1)

    x_test_1 = x_test.reshape(x_test.shape[0], -1)


    train_total_data = numpy.concatenate((x_train_1, y_train), axis=1)
    train_size = train_total_data.shape[0]



    validation_data = x_test_1
    validation_labels = y_test
    test_data = x_test_1
    test_labels = y_test

    return train_total_data, train_size, validation_data, y_train, test_data, test_labels, x_test



