import tensorflow.contrib.layers as lays
import tensorflow as tf
import os, sys
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

""" Tips """
# Use ptimizer='adadelta', loss='mean_squared_error'
# NUM_EPOCHS = 50, BATCH_SIZE = 512

'''
How to generate input for this encoder?
1. Generate the layouts for all the unique UI's using generate_layout_rico_dataset.py which uses .json file.
2. These layout images are the inputs to this gan.
'''

BATCH_SIZE = 50
epoch_num = 50     # Number of epochs to train the network
lr = 0.001        # Learning rate
img_shape = (120, 120, 3) 

def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [(img_shape[0]/4) * (img_shape[1]/4) * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, (img_shape[0]/4) * (img_shape[1]/4) * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

'''
Computation ->
(960, 540, 3) = 1555200 -> resize -> (1920, 1080, 3) = 6220800
z_dim = 100
[-1, 100] x [100, 6220800] -> [-1, 6220800] -> [-1, 1920, 1080, 3]
[-1, 1920, 1080, 3] x [3, 3, 1, 50] (strides = [1,2,2,1]) -> [-1, 960, 540, 50] -> reisze -> [-1, 1920, 1080, 50]
[-1, 1920, 1080, 50] x [3, 3, 50, 25] (strides = [1,2,2,1]) -> [-1, 960, 540, 25] -> resize -> [-1, 1920, 1080, 25]
[-1, 1920, 1080, 25] x [1, 1, 25, 1] (strides = [1,2,2,1]) -> [-1, 960, 540, 3]
'''
def generator(z, batch_size, z_dim):
    pixels = img_shape[0] * img_shape[1] * img_shape[2]
    g_w1 = tf.get_variable('g_w1', [z_dim, 4*pixels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [4*pixels], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 2*img_shape[0], 2*img_shape[1], img_shape[2]])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 3, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [2*img_shape[0], 2*img_shape[1]])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [2*img_shape[0], 2*img_shape[1]])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 3], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)
    
    # Dimensions of g4: BATCH_SIZE x img_shape[0] x img_shape[1] x 3
    return g4

def shuffle_train_data(train_filepaths):
    # shuffle labels and features
    idxs = np.arange(len(train_filepaths))
    np.random.shuffle(idxs)
    train_filepaths = np.array(train_filepaths)[idxs]

def data_iterator(num_train):
    """ A simple data iterator """
    batch_idx = 0
    for batch_idx in range(0, num_train, BATCH_SIZE):
        files_batch = train_filepaths[batch_idx:batch_idx+BATCH_SIZE]
        yield files_batch

def load_files(files_batch_val):
    w, h = img_shape[0], img_shape[1]
    input_data = np.zeros((len(files_batch_val),w,h,3))
    for i in range(len(files_batch_val)):
        img = cv2.imread(files_batch_val[i])
        #img = cv2.resize(img,(540, 960), interpolation = cv2.INTER_CUBIC)
        diff = (img.shape[0] - img.shape[1])/2
        img = cv2.copyMakeBorder(img, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=[255,255,255])
        img = cv2.resize(img,(w, h), interpolation = cv2.INTER_CUBIC)
        input_data[i,...] = img
    return input_data 

def generate_train_test_data(file, dataset_path):
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

    train_filepaths = [ dataset_path + fp for fp in training_list]
    test_filepaths = [ dataset_path + fp for fp in test_list]

    num_train = len(train_filepaths)
    return train_filepaths, test_filepaths

#######################################
# __main__ starts here
#######################################

tf.reset_default_graph()
file = sys.argv[1]
dataset_path = sys.argv[2]  #"/home/nkumarrai/sem_3/cse524/rico-dataset/layout/"

train_filepaths, test_filepaths = generate_train_test_data(file, dataset_path)

print "BATCH_SIZE %d and epochs %d"%(BATCH_SIZE,epoch_num)
iter_ = data_iterator(len(train_filepaths))
sess = tf.Session()

#######################################
# GAN network code starts here
#######################################
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None, img_shape[0], img_shape[1], img_shape[2]], name='x_placeholder') 
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, BATCH_SIZE, z_dimensions) 
# Gz holds the generated images

Dx = discriminator(x_placeholder) 
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, BATCH_SIZE, z_dimensions)
values_for_tensorboard = discriminator(x_placeholder) 
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

#############################################
# Initialize the network and start training.
#############################################

sess.run(tf.global_variables_initializer())
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, z_dimensions])
    try:
        files_batch_val = iter_.next()
        batch_images = load_files(files_batch_val)
        print "train batch_images.shape", batch_images.shape, " iteration in pre-train discriminator", i
    except StopIteration:
        print "Something is wrong, this loop had a small run, couldn't exhaust an epoch!"
        break

    __, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: batch_images, z_placeholder: z_batch})

    if(i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

for ep in range(epoch_num):  # epochs loop
    # Reinitialize the iterator again
    iter_ = data_iterator(len(train_filepaths))
    shuffle_train_data(train_filepaths)
    print "Epoch", ep, "started at time", str(datetime.now())
    count = 0
    while True:
        try:
            files_batch_val = iter_.next()
            batch_images = load_files(files_batch_val)
            count += 1
            #print "train batch_images.shape", batch_images.shape
            print "Epoch", ep, " images processed", count * BATCH_SIZE  
        except StopIteration:
            print "Current epoch is done, data was processed in BATCH_SIZE %d" % BATCH_SIZE
            break

        z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, z_dimensions])
        # Train discriminator on both real and fake images
        __, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: batch_images, z_placeholder: z_batch})

        # Train generator
        z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, z_dimensions])
        __ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

        if (count * BATCH_SIZE) % 1000 == 0:
            # Update TensorBoard with summary statistics
            z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, z_dimensions])
            summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: batch_images})
            writer.add_summary(summary, count)

            print("dLossReal:", dLossReal, "dLossFake:", dLossFake)
            # Every 100 iterations, show a generated image
            print("Iteration:", i, "at", str(datetime.now()))
            z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
            generated_images = generator(z_placeholder, 1, z_dimensions)
            images = sess.run(generated_images, {z_placeholder: z_batch})

            tmp = images[0].reshape([img_shape[0], img_shape[1], img_shape[2]])
            filename = "pretrained-model/image_" + str(ep) + "_" + str(count) + ".png"
            plt.imsave(filename, tmp, cmap='Greys') 
            print("image saved at %s" % filename)
 

            # Show discriminator's estimate
            im = images[0].reshape([1, img_shape[0], img_shape[1], img_shape[2]])
            result = discriminator(x_placeholder)
            estimate = sess.run(result, {x_placeholder: im})
            print("Estimate:", estimate)
    

# test the trained network
files_for_testing = test_filepaths[:100]
batch_images = load_files(files_for_testing)
print "test images.shape", batch_images.shape

# Save the variables to disk.
save_path = saver.save(sess, "rico_dataset_gan_model.ckpt")
print("Model saved in file: %s" % save_path)
