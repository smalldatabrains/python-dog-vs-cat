# python-dog-vs-cat
CNN and DNN to classify dog and cats pictures. Implementation is done with Tensorflow version from early 2017.

```imageprocessing.py``` is used to clean and format pictures of cats and dogs to a same standard.

```dnn.py``` aim to train a simple feed forward neural network.

```cnn.py``` and ```cnn2.py``` and ```cnn3.py``` are different attempt and architecture to train convolutional neural network.

I made this project in order to discover Tensorflow library. I could code Deep Neural Network from scratch quite easily. The design of the network can be changed when we declare it. Still, the prediction rate for cat and dogs qualification remains very low (around 65%) with this kind of network.
It finally turns out that I could achieve much better performance with CNN : 98% accuracy on training set and 82% on testing set.
 
