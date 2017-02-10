#dependencies
import os
import tensorflow as tf
import numpy as np
from numpy import array
from PIL import Image

#network parameters
learning_rate=0.08

#layers parameters
img_size=120
flat_img_dim=img_size*img_size

#different elements of the convolutional neural network --------------------------------------------
def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def create_bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))

def new_conv_layer(input,num_input_channels,filter_size,num_output_channels,use_pool=True):
	shape=[filter_size,filter_size,num_input_channels,num_output_channels]
	weights=create_weights(shape=shape)
	bias=create_bias(length=num_output_channels)
	layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')+bias
	if use_pool:
		layer=tf.nn.max_pool(layer,ksize=[1,4,4,1],strides=[1,2,2,1],padding='SAME')
	return layer,weights

def new_flatten_layer(layer):
	layer_shape=layer.get_shape()
	dimension=layer_shape[1:4].num_elements()
	layer=tf.reshape(layer,[-1,dimension])
	return layer,dimension

def fc_layer(layer,num_input,num_output,use_relu=True):
	weights=create_weights(shape=[num_input,num_output])
	bias=create_bias(length=num_output)
	layer=tf.matmul(layer,weights)+bias
	if use_relu:
		layer=tf.nn.relu(layer)
	return layer,weights

#initializing our network : 2 Conv layers, 1 flatten, 2 fully connected layer ----------------------

X=tf.placeholder(dtype=tf.float32,shape=[None,flat_img_dim],name="X")

Xm=tf.reshape(X,shape=[-1,img_size,img_size,1])

conv_layer1,conv_weights1=new_conv_layer(input=Xm,num_input_channels=1,filter_size=4,num_output_channels=10,use_pool=True)
conv_layer2,conv_weights2=new_conv_layer(input=conv_layer1,num_input_channels=10,filter_size=4,num_output_channels=8,use_pool=True)

flatten_layer,dimension=new_flatten_layer(layer=conv_layer2)

fc_layer1,fc_weights1=fc_layer(layer=flatten_layer,num_input=dimension,num_output=124)

fc_layer2,fc_weights2=fc_layer(layer=fc_layer1,num_input=124,num_output=2)

Yth=tf.nn.softmax(fc_layer2)
classes=tf.argmax(Yth,dimension=1)
Yreal=tf.placeholder(dtype=tf.int32,shape=[None,2],name="Yreal")

#cost and minimisation functions--------------------------------------------------------------------

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yth,labels=Yreal))
tf.summary.scalar('cross entropy',cross_entropy) #for tensorboard
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#Preparing training and testing set-----------------------------------------------------------------
train="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow\\greytrain" #folder with my 120x120 images
os.chdir(train)
list=os.listdir()
input=[]
output=[]
y_dog=[0,1] #dog vector
y_cat=[1,0] #cat vector
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
input=(input-np.mean(input,axis=0))/np.std(input,axis=0) #input normalization element-wise
output=np.asarray(output)
print(input.shape) #check shapes
print(output.shape)
#preparing training an testing set
permutation = np.random.permutation(input.shape[0]) #shuffling the rows
input=input[permutation]
output=output[permutation]
train_set=input[0:int(0.8*input.shape[0])]
train_label=output[0:int(0.8*input.shape[0])]
test_set=input[int(0.8*input.shape[0])+1:input.shape[0]]
test_label=output[int(0.8*output.shape[0])+1:output.shape[0]]

#session launch-------------------------------------------------------------------------------------
with tf.Session() as sess:
	merged=tf.summary.merge_all()
	writer=tf.train.SummaryWriter('C:/Users/Nicolas/Google Drive/website/python-tensorflow/src/graph',sess.graph)
	sess.run(tf.global_variables_initializer())
	print('Training started...')
	for epoch in range(15000):
		permutation=np.random.permutation(19600)
		permutation=permutation[0:50]
		batch=[train_set[permutation],train_label[permutation]]
		summary,model,cost=sess.run([merged,train_step,cross_entropy],feed_dict={X:batch[0],Yreal:batch[1]})
		writer.add_summary(summary,epoch)
		print(cost)
	print('Training finished.')