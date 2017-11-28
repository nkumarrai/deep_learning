import tensorflow.contrib.layers as lays
import tensorflow as tf
import os, sys
import numpy as np
from datetime import datetime
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

""" Tips """
# Use ptimizer='adadelta', loss='mean_squared_error'
# NUM_EPOCHS = 50, BATCH_SIZE = 512

'''
How to generate input for this encoder?
1. Generate the layouts for all the unique UI's using generate_layout_rico_dataset.py which uses .json file.
2. These layout images are the inputs to this gan.
'''

BATCH_SIZE = 500
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate

file = sys.argv[1]

input_list = []
with open(file, 'r') as f:
	with open(file, 'r') as f:
		for line in f:
			line = line.split('\n')[0]
			input_list.append(line)

print len(input_list), input_list[-1]

length_all = len(input_list)
training_size = int(0.8*length_all)

import random
training_ids = random.sample(range(0, length_all), training_size)

from operator import itemgetter 
training_list = (itemgetter(*training_ids)(input_list))
training_list = list(training_list)
print len(training_list), type(training_list)
input_list = set(input_list)
test_list = input_list - set(training_list)
test_list = list(test_list)
print len(test_list), type(test_list)

dataset_path = "/home/nkumarrai/sem_3/cse524/rico-dataset/layout/"
train_filepaths = [ dataset_path + fp for fp in training_list]
test_filepaths = [ dataset_path + fp for fp in test_list]

num_train = len(train_filepaths)

def shuffle_train_data(train_filepaths):
	# shuffle labels and features
	idxs = np.arange(len(train_filepaths))
	np.random.shuffle(idxs)
	train_filepaths = np.array(train_filepaths)[idxs]

def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        for batch_idx in range(0, num_train, BATCH_SIZE):
            files_batch = train_filepaths[batch_idx:batch_idx+BATCH_SIZE]
            yield files_batch

def load_files(files_batch_val):
    w, h = 960, 540
    input_data = np.zeros((BATCH_SIZE,w,h,3))
    for i in range(len(files_batch_val)):
        img = cv2.imread(files_batch_val[i])
        img = cv2.resize(img,(h, w), interpolation = cv2.INTER_CUBIC)
        input_data[i,...] = img
    return input_data 

# initialize the network
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()
print "BATCH_SIZE %d and epochs %d"%(BATCH_SIZE,epoch_num)
iter_ = data_iterator()
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        count = 0
        shuffle_train_data(train_filepaths)
        print "Epoch", ep, "started at time", str(datetime.now())
        while True:
            files_batch_val = iter_.next()
            batch_images = load_files(files_batch_val)
            print "train batch_images.shape", batch_images.shape
            count = count + batch_images.shape[0]
            print batch_images.shape, "count", count
            if count >= num_train:
                break

    # test the trained network
    files_for_testing = test_filepaths[:100]
    batch_images = load_files(files_for_testing)
    print "test images.shape", batch_images.shape
    # Save the variables to disk.
    # save_path = saver.save(sess, "model.ckpt")
    # print("Model saved in file: %s" % save_path)

