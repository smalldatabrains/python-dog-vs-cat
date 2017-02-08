#redim Images to a unique dimension (the smallest one from the dataset)

import os
import matplotlib
import numpy as np
from numpy import array
from PIL import Image,ImageOps
import PIL


#transform to grey images
def togrey(list):
	for file in list:
		img=Image.open(file)
		img=img.convert('L')
		img.save('C:/Users/Nicolas/Google Drive/website/python-tensorflow/greytrain/'+file,"JPEG")

#set same luminance on images

#Resize Images
def tothumb(list):
	dim=[]
	for file in list:
		img=Image.open(file)
		dim.append(img.size)
	
	avg=min(dim)
	
	for file in list:
		img=Image.open(file)
		img.thumbnail([120,120],Image.ANTIALIAS)
		img.save('C:/Users/Nicolas/Google Drive/website/python-tensorflow/thumbtrain/' + file,"JPEG")

def tocrop(list):
        dim=[]
        for file in list:
                img=Image.open(file)
                min=np.min(img.size)
                img=ImageOps.fit(img,(min,min),Image.ANTIALIAS)
                img.save('C:/Users/Nicolas/Google Drive/website/python-tensorflow/croptrain/' + file,"JPEG")

#it is better to thumb first and then crop (less loss of data)
# data="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow\\data"
# os.chdir(data)
# list=os.listdir()
# list.sort          
	
# tocrop(list)

data="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow\\croptrain"
os.chdir(data)
list=os.listdir()
list.sort

tothumb(list)

data="C:\\Users\\Nicolas\\Google Drive\\website\\python-tensorflow\\thumbtrain"
os.chdir(data)
list=os.listdir()
list.sort

togrey(list)