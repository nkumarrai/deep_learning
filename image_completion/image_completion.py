# To run tensor board
# bazel run tensorboard -- --logdir path/to/logs

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

height = 28
width = 28
channels = 3
n_inputs = height * width

X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(inputs=X, filters=64,  kernel_size=[5, 5], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
print(conv1)

conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[3, 3], dilation_rate=(1,1), strides=(2,2), padding="same", activation=tf.nn.relu)
conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
print(conv2)
print(conv3)

conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], dilation_rate=(1,1), strides=(2,2), padding="same", activation=tf.nn.relu)
conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
d_conv1 = tf.layers.conv2d(inputs=conv6,   filters=256, kernel_size=[3, 3], dilation_rate=(2,2), strides=(1,1), padding="same", activation=tf.nn.relu)
d_conv2 = tf.layers.conv2d(inputs=d_conv1, filters=256, kernel_size=[3, 3], dilation_rate=(4,4), strides=(1,1), padding="same", activation=tf.nn.relu)
d_conv3 = tf.layers.conv2d(inputs=d_conv2, filters=256, kernel_size=[3, 3], dilation_rate=(8,8), strides=(1,1), padding="same", activation=tf.nn.relu)
d_conv4 = tf.layers.conv2d(inputs=d_conv3, filters=256, kernel_size=[3, 3], dilation_rate=(16,16), strides=(1,1), padding="same", activation=tf.nn.relu)
conv7 = tf.layers.conv2d(inputs=d_conv4, filters=256, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
conv8 = tf.layers.conv2d(inputs=conv7,   filters=256, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
print(conv4)
print(conv5)
print(conv6)
print(d_conv1)
print(d_conv2)
print(d_conv3)
print(d_conv4)
print(conv7)
print(conv8)

d_conv5 = tf.layers.conv2d_transpose(inputs=conv8, filters=128, kernel_size=[4, 4], strides=(0.5,0.5), padding="same", activation=tf.nn.relu)
#d_conv5 = tf.nn.conv2d_transpose(conv8, filter=[4, 4, 256, 128], strides=[1, 0.5, 0.5, 1], padding='SAME')
#conv9 = tf.layers.conv2d(inputs=d_conv5, filters=128, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
print(d_conv5)
#print(conv9)


#d_conv6 = tf.layers.conv2d_transpose(inputs=conv9, filters=64, kernel_size=[4, 4], strides=(0.5,0.5), padding="same", activation=tf.nn.relu)
#d_conv6 = tf.nn.conv2d_transpose(conv9, filter=[4, 4, 128, 64], strides=[1, 0.5, 0.5, 1], padding='SAME')
#conv10 = tf.layers.conv2d(inputs=d_conv6, filters=32, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)
#out = tf.layers.conv2d(inputs=conv10, filters=3, kernel_size=[3, 3], dilation_rate=(1,1), strides=(1,1), padding="same", activation=tf.nn.relu)

#print(d_conv6)
#print(conv10)
#print(out)

# pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], stridess=2)
# pool3_flat = tf.reshape(pool3, [-1, 14 * 14 * 8])

# fc1 = tf.layers.dense(inputs=pool3_flat, units=16, activation=tf.nn.relu, name="fc1", reuse=None)
# logits = tf.layers.dense(inputs=fc2, units=4, activation=tf.nn.relu)
# Y_proba = tf.nn.softmax(logits)

# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
# loss = tf.reduce_mean(xentropy)

# optimizer = tf.train.AdamOptimizer()
# training_op = optimizer.minimize(loss)

# correct = tf.nn.in_top_k(logits, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
