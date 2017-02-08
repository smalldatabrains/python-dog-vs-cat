#Building the classifier with Tensorflow 1

import os
import matplotlib
import tensorflow as tf
from numpy import array
import numpy as np
import matplotlib
from PIL import Image

sess = tf.InteractiveSession()

#import images and flatten it into a array (examples x flatten_pixels)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#parameters
learning_rate=0.5
training_iteration=30
batch_size=10
display_step=2

#deisgn of network
n_hidden=2 #qty of hidden layers
n_neurons=[2000,200] #neurons for each hidden layer

#generate network:
def network_shape(n_hidden,n_neurons):
	with tf.name_scope("Layers"):
		global W
		global b
		W=dict()
		b=dict()
		for i in range(1,n_hidden+2):
			if i==1:
				W[i]=Weights(784,n_neurons[i-1])
				b[i]=bias(n_neurons[i-1])
			elif i==n_hidden+1:
				W[i]=Weights(n_neurons[i-2],10)
				b[i]=bias(10)
			else:
				W[i]=Weights(n_neurons[i-2],n_neurons[i-1])
				b[i]=bias(n_neurons[i-1])
	return W,b


#Create my Weights matrices and Bias vector for each layer of the NN 
def Weights(k,l):
	with tf.name_scope("Weights"):
		W=tf.Variable(tf.truncated_normal([k,l],stddev=0.1),name='Weights')
	return W

def bias(l):
	with tf.name_scope("bias"):
		b=tf.Variable(tf.zeros([l]),name="bias")
	return b

#test of network creation
network=network_shape(n_hidden,n_neurons)

# #Output Calculation
def multilayer(X,W,b):
	Yth=X
	for layer in range(1,len(W)+1):
		if layer!=len(W):
			Yth=tf.nn.relu(tf.matmul(Yth,W[layer])+b[layer])
		else:
			Yth=tf.nn.softmax(tf.matmul(Yth,W[layer])+b[layer])
	return Yth

#placeholders for Images and labels
with tf.name_scope("Inputs"):
	X = tf.placeholder(tf.float32, [None, 784],name='X')
	Yreal = tf.placeholder(tf.float32, [None, 10],name='Yreal')


Yth=multilayer(X,W,b)

#Cross entropy
with tf.name_scope("loss"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yth, labels=Yreal))

#Gradient Descent
with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#Initialization
init=tf.global_variables_initializer()

#--------------------------------------------------------------------------------------------------

dogdir="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow"
os.chdir(dogdir)

writer=tf.summary.FileWriter("logs/",sess.graph)
#--------------------------------------------------------------------------------------------------

#session start
sess.run(init)


# x_batch=tf.train.batch(input,batch_size,enqueue_many=True,allow_smaller_final_batch=True)
# y_batch=tf.train.batch(output,batch_size,enqueue_many=True,allow_smaller_final_batch=True)

for epoch in range(2000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={X: batch_xs, Yreal: batch_ys})
	print("epoch" ,epoch ,"is being processed")
	print(W[1].eval())

	#Learning curves

correct_prediction = tf.equal(tf.argmax(Yth,1), tf.argmax(Yreal,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Yreal: mnist.test.labels}))
