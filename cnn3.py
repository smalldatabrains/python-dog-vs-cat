#dependencies
import os
import math
import tensorflow as tf
import numpy as np
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt

#network parameters
drop_value=tf.placeholder(tf.float32)
global_step=tf.Variable(0,trainable=False)
starting_learning_rate=0.1
learning_rate=tf.train.exponential_decay(starting_learning_rate,global_step,1000,0.96)
tf.summary.scalar('learning rate',learning_rate)

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
	layer=tf.nn.dropout(tf.nn.conv2d(input=input,filter=weights,strides=[1,2,2,1],padding='SAME')+bias,drop_value)
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

#Print weitghs--------------------------------------------------------------------------------------
def print_conv_weights(weights):
	w=sess.run(weights)
	w_min=np.min(w)
	w_max=np.max(w)
	num_filter=w.shape[3]
	num_grids=math.ceil(math.sqrt(num_filter))
	fig,axes=plt.subplots(num_grids,num_grids)
	for i, ax in enumerate(axes.flat):
		if i<num_filter :
			img=w[:,:,0,i]
			ax.imshow(img,vmin=w_min,vmax=w_max,interpolation='nearest',cmap='seismic')
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

#Print convolutional layer--------------------------------------------------------------------------
def print_conv_layer(layer,image):
	feed_dict = {X: image,drop_value:1}
	values = sess.run(layer, feed_dict=feed_dict)
	num_filters = values.shape[3]
	num_grids = math.ceil(math.sqrt(num_filters))
	fig, axes = plt.subplots(num_grids, num_grids)
	for i, ax in enumerate(axes.flat):
		if i<num_filters:
			img = values[0, :, :, i]
			ax.imshow(img, interpolation='nearest', cmap='binary')
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()


#initializing our network : 2 Conv layers, 1 flatten, 2 fully connected layer ----------------------

X=tf.placeholder(dtype=tf.float32,shape=[None,flat_img_dim],name="X")

Xm=tf.reshape(X,shape=[-1,img_size,img_size,1])

conv_layer1,conv_weights1=new_conv_layer(input=Xm,num_input_channels=1,filter_size=3,num_output_channels=18,use_pool=True)
conv_layer2,conv_weights2=new_conv_layer(input=conv_layer1,num_input_channels=18,filter_size=3,num_output_channels=32,use_pool=True)

flatten_layer,dimension=new_flatten_layer(layer=conv_layer2)

fc_layer1,fc_weights1=fc_layer(layer=flatten_layer,num_input=dimension,num_output=200)

fc_layer2,fc_weights2=fc_layer(layer=fc_layer1,num_input=200,num_output=2)

Yth=tf.nn.softmax(fc_layer2)
classes=tf.argmax(Yth,dimension=1)
Yreal=tf.placeholder(dtype=tf.int32,shape=[None,2],name="Yreal")

#cost and minimisation functions--------------------------------------------------------------------

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yth,labels=Yreal))
tf.summary.scalar('cross entropy',cross_entropy) #for tensorboard
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

#performance of the network-------------------------------------------------------------------------
correct_prediction=tf.equal(classes,tf.argmax(Yreal,dimension=1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

#save the model-------------------------------------------------------------------------------------
saver=tf.train.Saver()

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
	train_writer=tf.summary.FileWriter('C:/Users/Nicolas/Google Drive/website/python-tensorflow/src/graph/train',sess.graph)
	test_writer=tf.summary.FileWriter('C:/Users/Nicolas/Google Drive/website/python-tensorflow/src/graph/test')
	sess.run(tf.global_variables_initializer())
	print('Training started...')
	for epoch in range(10000):
		permutation=np.random.permutation(19600)
		permutation=permutation[0:200]
		batch=[train_set[permutation],train_label[permutation]]
		summary,model,cost=sess.run([merged,train_step,cross_entropy],feed_dict={X:batch[0],Yreal:batch[1],drop_value:1})
		train_writer.add_summary(summary,epoch)
		print(cost)
		if epoch%500==0:
			print(epoch)
			summary=sess.run(merged,feed_dict={X:test_set[0:300],Yreal:test_label[0:300],drop_value:1})
			test_writer.add_summary(summary,epoch)

	print('Training finished.')
	print_conv_weights(conv_weights1)
	for image in range(10):
		print_conv_layer(conv_layer1,[test_set[image,:]])
	saver.save(sess,"C://Users//Nicolas//Google Drive//website//python-tensorflow//src//dogcatmodeldrop")
	print('Model Saved')

	testfolder="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow\\imtest" #folder with my 120x120 images
	os.chdir(testfolder)
	list=os.listdir()
	test=[]
	for file in list :
		img=Image.open(file)
		arr=array(img)
		if arr.shape[0]==120:
			flat=arr.flatten()
			test.append(flat)
	test=np.array(test)
	test.astype('float32')
	test=(test-np.mean(test,axis=0))/np.std(test,axis=0) #input normalization element-wise

	print(test.shape) #check shapes

	run=sess.run(Yth,feed_dict={X:test,drop_value:1})
	print(run)

sess.close()
