---
permalink: /mlp-convnets-lab-guided/
---

This page contains the guided laboratory of the MLP-CNN topic for the Deep Learning course at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya.

Table of Contents:

- [A Neural Network Playground](#playground)
- [Basic NN example](#basic_nn)


<a name='playground'></a>
### A Neural Network Playground

A good play to start for getting familiarized with how NN work is [tensorflow's playground](http://playground.tensorflow.org)


<a name='basic_nn'></a>
### Basic NN example

Let's see a basic example on how to use Kerass for training a simple NN. The following example can be find whole in the MAI-DL directory of the cluster. Here we split it in parts to review it.

We will work with the MNIST dataset of hand-written digits.
```python
from __future__ import division
import keras
print 'Using Keras version', keras.__version__
from keras.datasets import mnist

#Load the MNIST dataset, already provided by Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Check sizes of dataset
print 'Number of train examples', x_train.shape[0]
print 'Size of train examples', x_train.shape[1:]
```

We need to adapt the data to the shape of the input layer. Since its a fully connected layer (i.e., Dense), we need to flatten the RGB image. We also normalize the input, and make sure its type is correct.
```python
#Adapt the data as an input of a fully-connected (flatten to 1D)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
```

The labels of the data have to be transformed to the syntax required by the classifier (i.e., softmax)
```python
#Adapt the labels to the one-hot vector syntax required by the softmax
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

```

We can now define the architecture of our neural network. The input layer must be coherent with the shape of the data. We add two fully connected layers, and the final softmax classifier. 
```python
#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation
#Two hidden layers
nn = Sequential()
nn.add(Dense(128,activation='relu',input_shape=(784,)))
nn.add(Dense(64,activation='relu'))
nn.add(Dense(64,activation='relu'))
nn.add(Dense(10, activation='softmax'))
```

To visualize the architecture we just defined we can use the following keras call
```python
#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
from keras.util import plot_model
plot_model(nn, to_file='nn.png', show_shapes=true)
```

After defining the model, we need to compile it before running. At this point we define which optimizer we wish to use (e.g., SGD), how is the loss computed (e.g., categorical cross-entropy) and which metric we use to evaluate the model (e.g., accuracy)
```python
nn.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
```

With the model compile, we can now launch the training procedure. At this point we may define the batch size, and the number of epochs to run.
```python
history = nn.fit(x_train,y_train,batch_size=128,epochs=20)
```

We can now evaluate the trained model on the test set.
```python
score = nn.evaluate(x_test, y_test, verbose=0)
```

And plot the results and information on the training procedure
```python
##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
#No validation accuracy in this example
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#No validation loss in this example
#plt.legend(['train', 'test'], loc='upper left')
plt.legend(['train'], loc='upper left')
plt.savefig('model_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('model_loss.pdf')
```

For more information, we can visualize the confusion matrix. This is helpful to know how is our model behaving, and how can it be improved.
```python
#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = nn.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print 'Analysis of results'
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
```

Finally, we can store the model. This can be either done in two files (json for the architecture, hdf5 for the weights), or in a single hdf5 file.
```python
#Saving model and weights
from keras.models import model_from_json
nn_json = nn.to_json()
with open('nn.json', 'w') as json_file:
    json_file.write(nn_json)
weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
nn.save_weights(weights_file,overwrite=True)

#Loading model and weights
json_file = open('nn.json','r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)
nn.load_weights(weights_file)
```
