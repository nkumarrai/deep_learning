
# coding: utf-8

# In[1]:


# To run tensor board
# bazel run tensorboard -- --logdir path/to/logs

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[2]:


sample_image_d = mnist.train.next_batch(1)
sample_image = sample_image_d[0]
y = sample_image_d[1]
print(sample_image.shape)
print(y)

sample_image = sample_image.reshape([28, 28])
plt.imshow(sample_image, cmap='Greys')


# Softmax regression model with a single linear layer.

# In[3]:


import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
saver = tf.train.Saver()
save_path = saver.save(sess, "mnist_classify_basic_model.ckpt")
print("Model saved in file: %s" % save_path)

# Restore the model and predict
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
# Before restoring
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
saver.restore(sess, "mnist_classify_basic_model.ckpt")
# After restoring
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

