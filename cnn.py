#Building the classifier with Tensorflow

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array
import numpy as np
import math
from PIL import Image

#folder of images
train="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow\\greytrain"
os.chdir(train)
list=os.listdir()
input=[]
output=[]
y_dog=[0,1]
y_cat=[1,0]

#building input array (examples, flatten image)
for file in list :
	img=Image.open(file)
	arr=array(img)
	if arr.shape[0]==120:
		input.append(arr)
		if "cat" in file:
			output.append(y_cat)
		elif "dog" in file:
			output.append(y_dog)

input=np.array(input)
input.astype('float32')
output=np.asarray(output)
print(input.shape)
print(output.shape)

#Initialization of Weights and biases
def Weight(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))

#convolutional layer creation
def convolutional_layer(input,input_channels,filter_size,num_filters,use_pooling=True):
	shape=[filter_size,filter_size,input_channels,num_filters]
	weights=Weight(shape=shape)
	biases=bias(length=num_filters)
	layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')+biases
	if use_pooling:
		layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	layer = tf.nn.relu(layer)
	return layer,weights

#flatten layer
def flatten_layer(conv_layer):
	shape=conv_layer.get_shape()
	num_features=np.array(shape[1:4],dtype=int).prod()
	layer_flat=tf.reshape(conv_layer,[-1,num_features])
	return layer_flat

#fully connected layer
def fc_layer(inpu,num_inputs,num_outputs,use_relu=True):
	weights=Weight(shape=[num_inputs,num_outputs])
	biases=bias(length=num_outputs)
	layer=tf.matmul(input,weights)+biases
	if use_relu:
		layer=tf.nn.relu(layer)
	return layer

#placeholders
X=tf.placeholder(tf.float32,shape=[-1,img_size,img_size,num_channels])
Yreal=tf.placeholder(tf.float32,shape=[None,2])

#cost function and gradient descent
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yth, labels=Yreal))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#---------------------------------------------------------------------------------------------------------------------
#convolutional Layer 1
filter_size1=5
num_filters1=16
#convolutional Layer 2
filter_size2=5
num_filters2=36

#fully connected layer
fc_size=124

#training session

#performance measurement
correct_prediction = tf.equal(tf.argmax(Yth,1), tf.argmax(Yreal,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction=Yth
print(sess.run(prediction, feed_dict={X: test_set, Yreal: test_label,keep_prob:1}))
print(sess.run(accuracy, feed_dict={X: test_set, Yreal: test_label,keep_prob:1}))
print(sess.run(accuracy, feed_dict={X: train_set, Yreal: train_label,keep_prob:1}))
sess.close()