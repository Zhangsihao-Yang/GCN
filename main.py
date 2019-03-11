from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
import h5py
import argparse
import os 
import sys
sys.path.insert(0, 'src')
from model import *
# from utils import * 

random_seed = 0
np.random.seed(random_seed)

sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=int, default=0)
parser.add_argument('--num_iterations', type=int, default=50000)
parser.add_argument('--num_input_channels', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--num_points', type=int, default=6146)
parser.add_argument('--results_path', default='results')

FLAGS = parser.parse_args()

ARCHITECTURE = FLAGS.architecture
NUM_ITERATIONS = FLAGS.num_iterations
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
LEARNING_RATE = FLAGS.learning_rate
NUM_POINTS = FLAGS.num_points
RESULTS_PATH = FLAGS.results_path

if not os.path.exists(RESULTS_PATH):
		os.makedirs(RESULTS_PATH)

"""
Load dataset 
x (train_data) of size [batch_size, num_points, in_channels] :
	in_channels can be x,y,z coordinates or any other descriptor
adj (adj_input) of size [batch_size, num_points, K] : 
	This is a list of indices of neigbors of each vertex. (Index starting with 1)
	K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.

"""
BATCH_SIZE = 1
IN_CHANNELS = 3
K = 4
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])
adj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])
# y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])

rec_x = get_ae_model(x, adj, ARCHITECTURE)

batch = tf.Variable(0, trainable=False)

# Standard classification loss
# cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), axis=1))
mse_loss = tf.losses.mean_squared_error(x, rec_x)

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(mse_loss, global_step=batch)
# correct_prediction = tf.equal(tf.argmax(y_conv,2), tf.argmax(y_,2))
# predictions = tf.argmax(y_conv, 2)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		write_logs("Checkpoint restored\n")

# Train for the dataset

input = np.load('../data/exampleVertex.npy')
input = np.expand_dims(input, 0)
train_adj = np.load('../data/adj.npy')
train_adj = np.expand_dims(train_adj, 0)

for iter in range(NUM_ITERATIONS):
	# i = train_shuffle[iter%(len(train_data))]
	# input = train_data[i]
	# if iter%1000 == 0:
		# train_accuracy = accuracy.eval(feed_dict={x:input, adj:adj_input, y_: label})
		# write_logs("Iteration %d, training accuracy %g\n"%(iter, train_accuracy))
	train_step.run(feed_dict={x:input, adj:train_adj})
	loss = mse_loss.eval(feed_dict={x:input, adj:train_adj})
	print('For iter {}, the loss is: {}'.format(iter, loss))
# To save the generated point and visualize it in meshlab



