import os
import time
import glob
import pickle
import random
import numpy as np
import scipy.io as sio
import scipy.stats as scst
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, f1_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

from scipy.fftpack import fft, ifft

def center_hip_at_zero(actions):
	batch_size = actions.shape[0]
	sequence_length = actions.shape[1]
	num_channels = actions.shape[2]
	num_ch = int(num_channels / 3)

	actions_reshape = np.reshape(actions, [batch_size, sequence_length, num_ch, 3])
	hip_location = np.expand_dims(actions_reshape[:,:,0,:], axis=2)
	hip_location_tile = np.tile(hip_location, [1,1,num_ch,1])
	actions_reshape = actions_reshape - hip_location_tile
	actions = np.reshape(actions_reshape, [batch_size, sequence_length, num_channels])

	return actions


def get_hmd(test_id=1, dataset_dir='HDM_data/'):

    ch = 93
    num_classes = 130


    if test_id < 1 or test_id > 5:
        test_id = 1

    fold = test_id
    all_data = sio.loadmat('HDM_data/HDM_data_fold_' + str(fold) +'.mat')
    train_data = all_data['train_data']/50.0
    train_label = all_data['train_label']
    test_data = all_data['test_data']/50.0
    test_label = all_data['test_label']

    num_train = train_data.shape[0]
    np.random.seed(0)

    valid_data = train_data[1500:,:,:]

    sequence_length = train_data.shape[1]
    num_channels = train_data.shape[2]


    seed_value = 1
    window_size = 100
    input_shape = [window_size,3]
    train_set_size = 1.0
    x = train_data
    y = train_label

    x_test = test_data
    y_test = test_label

  
    print("hmd data size ",train_data.shape, train_label.shape ,test_data.shape, test_label.shape)

    x_temp = x.reshape((-1,int(window_size),ch))
    x_train = x.reshape((-1,int(window_size),ch))
    x_test = x_test.reshape((-1,int(window_size),ch))

    train_set_size = int(x_train.shape[0])
    random.seed(seed_value)

    train_indices = random.sample(range(x_train.shape[0]),train_set_size)

    x_train = x_train[train_indices,:,:]
    y_train = y[train_indices,:]


    print('-------------GENE train test data-----------------')

    print('Train data shape: ',x_train.shape)
    print('Test data shape: ',x_test.shape)
    print('Train label shape: ',y_train.shape)
    print('Test label shape: ',y_test.shape)




    print('-------------Train data #-----------------')

    for idx in range(num_classes):
        print(idx, "-", (y_train[:,idx] == 1).sum()) #idx

    print('-------------Test data #-----------------')
    for idx in range(num_classes):
        print(idx, "-", (y_test[:,idx] == 1).sum())


    x_train1 = x_train
    x_test1 = x_test

    print("max3b ",x_test.max()) #1
    print("min3b ",x_test.min()) #0


    x_train = center_hip_at_zero(x_train1)

    x_test = center_hip_at_zero(x_test1)

    print("max",(x_train.max()))
    print("min",(x_train.min()))

    x_train = (x_train + 1) / 2.0
    x_test = (x_test + 1) / 2.0    
    
    return x_train, y_train, x_test, y_test




if __name__ == "__main__":
    print("get_hmd")

    a, b, c, d = get_hmd(1)
    a, b, c, d = get_hmd(2)
    a, b, c, d = get_hmd(3)
    a, b, c, d = get_hmd(4)
    a, b, c, d = get_hmd(5)


    print("---"*20)

