#Building the classifier with Tensorflow

import os
import csv
import matplotlib
import tensorflow as tf
from numpy import array
import numpy as np
from PIL import Image

#import images and flatten it into a array (examples x flatten_pixels)

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
		flat=arr.flatten()
		input.append(flat)
		if "cat" in file:
			output.append(y_cat)
		elif "dog" in file:
			output.append(y_dog)

input=np.array(input)
input.astype('float32')

input=(input-np.mean(input,axis=0))/np.std(input,axis=0)

output=np.asarray(output)
print(input.shape)
print(output.shape)

#design of network
n_neurons=[124] #neurons for each hidden layer
n_hidden=len(n_neurons)

#generate network:
def network_shape(input,ouput,n_hidden,n_neurons):
	with tf.name_scope("Layers"):
		global W
		global b
		W=dict()
		b=dict()
		for i in range(1,n_hidden+2):
			if i==1:
				W[i]=Weights(input.shape[1],n_neurons[i-1])
				b[i]=bias(n_neurons[i-1])
			elif i==n_hidden+1:
				W[i]=Weights(n_neurons[i-2],output.shape[1])
				b[i]=bias(output.shape[1])
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
		b=tf.Variable(tf.truncated_normal([l],stddev=0.1),name="bias")
	return b

#test of network creation
network=network_shape(input,output,n_hidden,n_neurons)

# #Output Calculation
def multilayer(X,W,b):
	Yth=X
	for layer in range(1,len(W)+1):
		if layer!=len(W):
			Yth=tf.nn.dropout(tf.nn.relu(tf.matmul(Yth,W[layer])+b[layer]),keep_prob)
		else:
			Yth=tf.nn.softmax(tf.matmul(Yth,W[layer])+b[layer])
	return Yth

#placeholders for Images and labels
with tf.name_scope("Inputs"):
	X = tf.placeholder(tf.float32, [None,input.shape[1]],name='X')
	Yreal = tf.placeholder(tf.float32, [None,output.shape[1]],name='Yreal')
	global_step=tf.Variable(0,trainable=False)
	keep_prob = tf.placeholder(tf.float32)


starter_learning_rate=0.07
learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,100,0.9)

Yth=multilayer(X,W,b)


#Cross entropy
with tf.name_scope("loss"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yth, labels=Yreal))

#Gradient Descent
with tf.name_scope("train"):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

#Initialization
init=tf.global_variables_initializer()

#--------------------------------------------------------------------------------------------------
#buidling the logs of the graph for visualization in tensorboard
dogdir="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow"
os.chdir(dogdir)

#--------------------------------------------------------------------------------------------------

#separate training set (80% of data) and testing set (20% of data)

permutation = np.random.permutation(input.shape[0]) #shuffling the rows
input=input[permutation]
output=output[permutation]
train_set=input[0:int(0.8*input.shape[0])]
train_label=output[0:int(0.8*input.shape[0])]
test_set=input[int(0.8*input.shape[0])+1:input.shape[0]]
test_label=output[int(0.8*output.shape[0])+1:output.shape[0]]

#session start
sess = tf.Session()
sess.run(init)

for epoch in range(0,10000):
	permutation=np.random.permutation(19600)
	permutation=permutation[0:100]
	batch=[train_set[permutation],train_label[permutation]]
	training,loss_val=sess.run([train_step,cross_entropy],feed_dict={X:batch[0],Yreal:batch[1],keep_prob:0.75})
	print("epoch" ,epoch ,"is being processed")
	print(loss_val,W[1].eval(session=sess))


#accuracy on the test set
correct_prediction = tf.equal(tf.argmax(Yth,1), tf.argmax(Yreal,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction=Yth
print(sess.run(prediction, feed_dict={X: test_set, Yreal: test_label,keep_prob:1}))
print(sess.run(accuracy, feed_dict={X: test_set, Yreal: test_label,keep_prob:1}))
print(sess.run(accuracy, feed_dict={X: train_set, Yreal: train_label,keep_prob:1}))
sess.close()