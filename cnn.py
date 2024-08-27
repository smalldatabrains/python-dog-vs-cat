#Building the classifier with Tensorflow

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array
import numpy as np
import math
from PIL import Image

#folder of images
train="img/greytrain/"
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

#preparing training an testing set
permutation = np.random.permutation(input.shape[0]) #shuffling the rows
input=input[permutation]
output=output[permutation]
train_set=input[0:int(0.8*input.shape[0])]
train_label=output[0:int(0.8*input.shape[0])]
test_set=input[int(0.8*input.shape[0])+1:input.shape[0]]
test_label=output[int(0.8*output.shape[0])+1:output.shape[0]]

#parameters
img_size=120
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_channels=1
num_classes=2

filter_size1 = 5   
num_filters1 = 4        
filter_size2 = 5      
num_filters2 = 10
fc_size = 128       

#Initialization of Weights and biases
def Weight(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))

#convolutional layer
def convolutional_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
	shape=[filter_size,filter_size,num_input_channels,num_filters]
	weights=Weight(shape=shape)
	biases=bias(length=num_filters)
	layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,2,2,1],padding='SAME')+biases
	if use_pooling:
		layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	layer = tf.nn.relu(layer)
	return layer,weights

#flatten layer
def flatten_layer(layer):
	layer_shape=layer.get_shape()
	num_features=layer_shape[1:4].num_elements()
	layer_flat=tf.reshape(layer,[-1,num_features])
	return layer_flat,num_features

#fully connected layer
def fc_layer(input,num_inputs,num_outputs,use_relu=True):
	weights=Weight(shape=[num_inputs,num_outputs])
	biases=bias(length=num_outputs)
	layer=tf.matmul(input,weights)+biases
	if use_relu:
		layer=tf.nn.relu(layer)
	return layer

#placeholders
X=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='X')
X_image=tf.reshape(X,[-1,img_size,img_size,num_channels])
Yreal=tf.placeholder(tf.float32,shape=[None,2],name='Yreal')
Yreal_cls=tf.argmax(Yreal,dimension=1)

#layers creation
layer_conv1, weights_conv1 = \
    convolutional_layer(input=X_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    convolutional_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

Yth = tf.nn.softmax(layer_fc2)
Yth_cls = tf.argmax(Yth, dimension=1)

#cost function and gradient descent
learning_rate=0.05
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=Yreal))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#---------------------------------------------------------------------------------------------------------------------
#performance measurement
correct_prediction = tf.equal(Yth_cls, Yreal_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#training session
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for epoch in range(0,15000):
	permutation=np.random.permutation(19600)
	permutation=permutation[0:50]
	batch=[train_set[permutation],train_label[permutation]]
	training,loss_val=sess.run([train_step,cross_entropy],feed_dict={X:batch[0],Yreal:batch[1]})
	print("epoch" ,epoch ,"is being processed")
	print(loss_val)

print(sess.run(accuracy, feed_dict={X: train_set, Yreal: train_label}))
print(sess.run(accuracy, feed_dict={X: test_set, Yreal: test_label}))
sess.close()