#Building the classifier with Tensorflow

import os
import matplotlib
import tensorflow as tf
from numpy import array
import numpy as np
from PIL import Image

#Initialization of Weights and biases
W=tf.Variable(tf.truncated_normal([fenx,feny,input_channels,output_channels],stddev=0.1))
b=tf.Variable(tf.ones([output_channels]/10))
init=tf.global_variables_initializer()

#placeholders
X=tf.placeholder()
Yth=tf.placeholder()
pkeep=tf.placeholder(tf.float32)

#Forward propagation

Yconv=tf.nn.relu(tf.nn.conv2d(X,W,strides[1,x_stride,y_stride,1],padding='SAME')+b)	
Yfull=tf.reshape(Yconv,shape=[-1,full_size])
Ythe=tf.nn.relu(tf.matmul(Yfull,W)+b)
Yth=tf.nn.softmax(tf.matmul(Ythe,W)+b)

#cost function and gradient descent

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yth, labels=Yreal))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

#training session