import tensorflow.contrib.layers as lays
import tensorflow as tf
import os, sys
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

""" Tips """
# Use ptimizer='adadelta', loss='mean_squared_error'
# NUM_EPOCHS = 50, BATCH_SIZE = 512

'''
How to generate input for this encoder?
1. Generate the layouts for all the unique UI's using generate_layout_rico_dataset.py which uses .json file.
2. Layout -> resize it to 100x56, separate out texts and non-texts boxes in two b/w images.
3. Vectorize the two images in a single vector of size (11200, 1).
4. Save these vectors in .txt files. 
5. This encoder processes these .txt files.
'''

BATCH_SIZE    = 500
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate

file = sys.argv[1]
# input_list = []
# count_list = []
# with open(file, 'r') as f:
# 	for line in f:
# 		line = line.split('\n')[0]
# 		if line.endswith('_input.txt'):
# 			input_list.append(line)
# 		elif line.endswith('_count.txt'):
# 			count_list.append(line)

# print len(input_list), len(count_list)

# with open('layout_list_input.txt' , 'w') as finput:
# 	with open('layout_list_element_count.txt', 'w') as felement:
# 		for i,c in zip(input_list, count_list):
# 			finput.write(i + '\n')
# 			felement.write(c+ '\n')

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

# training_list = [input_list[i] for i in training_ids]
# test_list = [input_list[i] for i in range(length_all) if i not in training_ids]
# print len(training_list), len(test_list), len(training_list) + len(test_list)

from operator import itemgetter 
training_list = (itemgetter(*training_ids)(input_list))
training_list = list(training_list)
print len(training_list), type(training_list)
input_list = set(input_list)
test_list = input_list - set(training_list)
test_list = list(test_list)
print len(test_list), type(test_list)

dataset_path = "/home/nkumarrai/sem_3/cse524/rico-dataset/generated_layout/"
train_filepaths = [ dataset_path + fp for fp in training_list]
test_filepaths = [ dataset_path + fp for fp in test_list]

num_train = len(train_filepaths)

def shuffle_train_data(train_filepaths):
	# shuffle labels and features
	idxs = np.arange(len(train_filepaths))
	print idxs
	np.random.shuffle(idxs)
	print idxs
	train_filepaths = np.array(train_filepaths)[idxs]

def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        for batch_idx in range(0, num_train, BATCH_SIZE):
            files_batch = train_filepaths[batch_idx:batch_idx+BATCH_SIZE]
            yield files_batch

def load_files(files_batch_val):
	input_data = np.fromfile(files_batch_val[0], dtype=int, count=-1, sep='\n')
	for i in range(1, len(files_batch_val)):
		matf = np.fromfile(files_batch_val[i], dtype=int, count=-1, sep='\n')
		input_data = np.vstack((input_data, matf))
	return input_data


def autoencoder(inputs):
    # encoder
    # 11200	-> 2048
    # 2048 	-> 256
    # 256   -> 64
    net = tf.layers.dense(inputs=inputs, units=2048, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu)
    # decoder
    # 64	-> 256
    # 256	-> 2048
    # 2048	-> 11200
    net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, units=2048, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, units=11200, activation=tf.sigmoid)
    return net

ae_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 11200])
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

iter_ = data_iterator()
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
    	count = 0
    	shuffle_train_data(train_filepaths)
    	print "Epoch", ep, "started at time", str(datetime.now())
        while True:
            files_batch_val = iter_.next()
            features = load_files(files_batch_val)/255.0
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: features})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
            count = count + features.shape[0]
            print features.shape, "count", count
            if count >= num_train:
            	break

    # test the trained network
    files_for_testing = test_filepaths[:100]
    features = load_files(files_for_testing)
    test_out = sess.run([ae_outputs], feed_dict={ae_inputs: features})[0]
    np.savetxt("test_after_training.csv", test_out, delimiter=',',fmt='%d')
    np.savetxt("test_files.csv", files_for_testing, delimiter=',',fmt='%s')

	# Save the variables to disk.
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)

