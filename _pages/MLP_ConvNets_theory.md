---
permalink: /mlp-convnets-theory/
---

This page contains the theoretical part of the MLP-CNN topic for the Deep Learning course at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya. 

Table of Contents:

- [A Bit of History](#history)
- [Rosenblatt's Perceptron](#rosenblatt)
- [Backpropagation and Stochastic Gradient Descent](#backprop)
- [Activation Functions](#activations)
- [Training Parameters](#training_params)
    - [Epochs and Batch Size](#epochs_batch)
    - [Learning Rate](#learning_rate)
    - [Other Parameters](#other_params)
- [Adaptative Learning Methods](#adaptative_methods)
- [Convolutional Neural Networks](#cnn)
    - [CNN Parameters](#cnn_params)
    - [CNN Volumes](#cnn_volumes)
- [CNN Architectures](#conv_arch)
- [Regularization Methods](#regularization)
- [CNN from the Inside](#inside)
- [CNN Applications](#apps)
- [Bibliography](#bib)


This document contains an overview of the main components of ANN and CNNs. Most of these are only introduced, and references are given for the interested reader.


<a name='history'></a>
### A Bit of History

Neural networks have recently become a hot topic for both academia and industry. However, neural networks have been around for a while. The first contribution to Artificial Neural Networks (ANN) is considered to be a paper published by Warren McCulloch and Walter Pitts in 1943 [1]. In their work, McCulloch and Pitts tried to understand how could the brain compute highly complex behaviors, using processing units as simple as neurons. Their design of single neurons with several weighted inputs and an output tried to mimmick human neurons, and is still being used today.

<div style="text-align:center">
    <img src="/images/mcculloch_fig1.jpg" width="450">
</div>

<div><p style="text-align: center;">Original Figure from McCulloch and Pitts. Source [1].</p></div>

<a name='rosenblatt'></a>
### Rosenblatt's Perceptron

In 1958,Frank Rosenblatt developed the "Perceptron" algorithm [2], which was based on McCulloch and Pitts neurons. The algorithm was a binary classifier, mapping a real valued input to a single binary output.

$$
f(x)= \begin{cases}1\ \ if\ w \cdot x+b>0\\ 0\ otherwise \end{cases}
$$

Where w is is a vector of real-valued weights, · is the dot product, and b is a real scalar constant.

Rosenblatt implemented the algorithm within the machine "Mark I Perceptron", a visual classifier composed by 400 photosensitive receptors (sensory units), associated with 512 stepping motors (association units), and an output of 8 neurons (response units) [3]. It contained only one layer of trainable parameters (association units). For more details see [4,9].

<div style="text-align:center">
    <img src="/images/Mark1.png" width="500">
</div>
<p style="text-align: center;">Illustration of the Mark I Perceptron from [3].</p>

Rosenblatt acknolwedged a set of limitations of his Perceptron machine in a series of publications. At the same time, Minsky and Papert published the book "Perceptrons: an introduction to computational geometry", also detailing some of those limitations. See [6] for more details. 
The work of Minsky and Papert had a huge impact on the public, although few people actually understood the nature of their contribution. Simply put, Minsky and Papert argue that a multilayer network (or MLP) is needed for learning certain basic functions such as XOR, while no appropriate training algorithm existed for that kind of network at the time.

Regardless of the technical aspects of Misnky and Papert's work, the reaction from the public was a drastic cut in funding on ANN during the 70s, until the mid 80s, in what is knows as the "AI Winter". After ANNs were almost abandoned, AI research focused on "expert systems" instead, which would also suffer their own "AI Winter" in the 90s.

<a name='backprop'></a>
### Backpropagation and Stochastic Gradient Descent

The backpropagation algorithm reignited the interest on ANN. Originally intended for training multi-layer ANN by Webos in 1974 [7], it gained attention when rediscovered by Rumelhart, Hinton and Williams in 1985 [8], and effectively finished the "AI Winter" on ANN.

For training a multi-layer ANN, the backpropagation starts with a forward pass of an input. After the input has reached the last layer of the network, the label predicted by the ANN (the network output) is compared against the ground truth label, and the error made by the network is measured using a loss function. The computed loss is used to update the weights of the last layer of the network, finding the best gradient towards minimizing the error (though a partial derivative). The challenge in training multi-layer ANN resides in the complexity of optimizing the weights of layers not directly connected to the output. To solve that, the backpropagation algorithm uses the chain rule, which allows to compute the derivative of previous layers, and thus update the weights of those layers as well. Simply put, the backpropagation algorithm is capable of determining the responsability of each individual neuron for the error made by the network, regardless of the neuron location within the net. Traditionally, the change in the weights has been computed using an optimization algorithm called Stochastic Gradient Descent (SGD). SGD iteratively estimates the best direction for optimization using a subset of the whole dataset (hence, stochastic). Since then, several alternatives have been proposed. Some of these alternatives are listed later in this document in the [Adaptative Learning Methods](#adaptative_methods) section.   

By applying the backpropagation process iteratively (forward pass of the input, calculate error made through loss function, backpropagate prediction error layer by layer, update neuron weights through optimization), we can eventually fit the weights of all layers in a MLP for a given problem. See [10] for a full mathematical explanation of the backprop algorithm, [42] gives simpler visual explanation, and [43] shows why it is important to understand how it works with practical examples.


With this new training methodology, research on ANN became active again. LeCun et. al. [11] developed a digit recognition system using data from the US Postal Service in 1989, and showed how ANN could be used to solve complex practical problems. LeCun's system included a layer of convolutional neurons, which had been previously proposed by Fukushima in 1980 [12].

<a name='activations'></a>
### Activation Functions

The biologically inspired ANN use an activation function for every neuron, which determines the neuron's output given the weighted sum of its inputs. Rosenblatt's Perceptron used the simplest one (Equation 1), a binary function that either activates with the value, or it does not. Nowadays more complex functions are used, all of them with a slope of some sort. The activation function needs to be non-linear, to provide the ANN with the capacity to learn non-linear patterns. 

The most popular options are:

The Sigmoid: $f(x)=\frac{1}{1+e^{-x}}$
<div style="text-align:center">
    <img src="/images/sigmoid.png" width="350">
</div>
<p style="text-align: center;">Sigmoid function, from [19].</p>
 
The Tanh: $f(x)=\frac{2}{1+e^{-2x}}-1$
<div style="text-align:center">
    <img src="/images/tanh.png" width="350">
</div>
<p style="text-align: center;">Tanh function, from [19].</p>

 The Rectified Linear Unit (ReLU): $f(x)=max(0,x)$
<div style="text-align:center">
    <img src="/images/relu.jpeg" width="350">
</div>
<p style="text-align: center;">ReLU function, from [19].</p>

ReLU are currently being used by default in most cases, as these are more efficient. ReLU also avoids the vanishing gradient problem through its constant slope (see [20] for more details on the vanishing gradient). For more details on activation functions, see [21].


<a name='training_params'></a>
### Training Parameters

While training a deep network, there are many training paremeters that can be tuned, and finding a good set is not straight-forward. In most cases, its just a matter of try and error, which typically requires executing the training procedure many times. Next we review some of those parameters.


<a name='epochs_batch'></a>
#### Epochs and batch size

An epoch correponds to a training stage where all images in the training set are used for training once. Typically, an ANN training will require lots of epochs, as each example takes the network parameters in a different direction, and the network can learn more than once from the same example given the changes produced by the other examples. In most cases,  when using too many epochs the network will eventually overfit, as the network will learn to memorize the inputs after seeing them too many times. Overfitting can be identified by a decresing loss on the training set (which indicates that the network is doing increasibly better on that piece of data) together with an increase in the validation or test set (which indicates that the network is doing worse on those parts of the dataset, hence it is not generalizing).

The batch size defines the number of input examples that are processed together in a single forward and backward pass of the network. In other words, is the number of inputs that the network 'sees together'. Batch size can go from 1 (one example at a time) to full dataset size. Larger batch sizes will train faster, but may be less precise [23]. The most commonly used batch sizes are 16 and 32.

Batch sizes too small may be identified by a noisy function of loss over time, as the loss on one batch differs significantly from the loss of the next batch. With a batch size equal to the dataset size, variation should be minimal (unless the learning rate is too high, in which case the network cannot converge).

<div style="text-align:center">
    <img src="/images/loss.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration of a noisy loss function, from [22].</p>



<a name='learning_rate'></a>
#### Learning Rate

Learning rate defines the amount of the total derivative of the loss that is applied on each step. That is, how much the weights will be changed towards the presumed optimal. Typically, applying the full derivative (learning rate of 1) provides an overcorrection, and smaller steps are required. A learning rate too large will cause the neurons to make large "jumps" within the high-dimensional space being defined, making them unable to converge. A learning rate too small may make training unfeasibly long, as the steps taken towards the optimal weights are too small. Its often better to start with a small learning rate to get weights in the right direction, and increase it later on to improve convergence.

<div style="text-align:center">
    <img src="/images/learningrates2.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration on the behavior of a large and small learning rate, from [40].</p>


Plotting the behavior of the loss over time gives a good insight on our choice of learning rate.

<div style="text-align:center">
    <img src="/images/learningrates.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration on the behavior of various learning rates, from [22].</p>



<a name='other_params'></a>
#### Other parameters

There are many other parameters that may affect the learning process. We will not detail them here, these include:

- Weight Decay: After each update, weights are multiplied by a factor between 0 and 1. This is a form of regularization.
- Weight initialization: The initial weight and bias values before training starts is key for the outcome of the training. Xavier initialization is currently a popular solution.
- Momentum: To help convergence, momentum adds a fraction of the previous weight update to the current weight update. This increases convergence if both updates were in the both direction, and smooths variations when updates go on different directions.

[22] contains several tips on how to optimize all these paremeters (Hyperparameter optimization Section).



<a name='adaptative_methods'></a>
### Adaptative learning methods

The number of parameters to be fit on learning methods like SGD (e.g., learning rate, momentum, weight decay, etc.) has motivated the appearence of learning methods which choose and adapt these parameters automatically. These are used instead of SGD, and can simplify the parameter tuning process significantly. Among the most popular alternatives, there is Adagrad, Adadelta, RMSprop and Adam. See [41] for more details on these.

<div style="text-align:center">
    <img src="/images/saddle_point_evaluation_optimizers.gif" width="450">
</div>
<p style="text-align: center;">Illustration on the behavior of various learning algorithms, from [41].</p>


<a name='cnn'></a>
### Convolutional Neural Networks

Convolutional Neural Networks (CNN) are based on a special type of neuron: convolutional neurons. Typically, convolutional neurons have a limited input, e.g., are only connected with a few neurons from the previous layer. In contrast, neurons from fully-connected layers are connected to all the neurons from the previous layer.

For defining their input connectivity, conv neurons assume the existence of a two-dimensional structure in the data. To capture spatially consistent patterns occuring in a subset of the input, conv neurons are connected to a set of neurons from the previous layer that define squared patches.

<div style="text-align:center">
    <img src="/images/convolution2.png" width="350">
</div>
<p style="text-align: center;">Illustration of a conv neuron, from [13]. The Kernel in the figure corresponds to the learnt weights of the red neuron. This neuron has 9 inputs (3x3) from the previous layer.</p>

Through this limited connectivity, conv layers can focus on a particular patch of the input. Significantly, the kernel (i.e., the set of weights) learnt for that patch may be relevant for all the other patches in the input. Hence, we can define similar neurons that use the same kernel but which focus on other parts of the input. This idea is commonly known as "weight sharing", as several neurons of the same layer are defined by a common set of weights, and allows for the use of lots of neurons using the same set of trainable parameters. 

The convolution weight sharing process considers the definition of several neurons with equal weights processing different parts of the input. A different way of understanding this process is to consider that a single filter is slided (or convolved) over all the input, to produce a set of outputs corresponding to the full input. See the next figure for a visual example.

<div style="text-align:center">
    <img src="/images/Convolution_schematic.gif" width="350">
</div>
<p style="text-align: center;">Illustration of a filter being convolved, from [14]. The 3x3 kernel weights are shown in red.</p>

Convolving filters over an input can have many effects. Averaging each pixel by its neighbors blurs the image, while taking the difference between the central pixel and the rest can be used to detect edges.


<div style="text-align:center">
   <img src="/images/convolution-blur.png" width="200"><img src="/images/generic-taj-convmatrix-blur.jpg" width="300"><br>
   <img src="/images/convolution-edge-detect1.png" width="200"><img src="/images/generic-taj-convmatrix-edge-detect.jpg" width="300">
</div>
<p style="text-align: center;">Top: Convolved filter to blur image. Bottom: Convolved filter to detect edges. From [15].</p>

<a name='cnn_params'></a>
#### CNN Parameters

The convolution process has several parameters. The kernel size (also known as receptive field) determines the patch of the image each filter will be able to process. Stride defines the movement of the kernel through the input, and padding defines what to do with the borders of the input.

* Stride: When applying the same convolving filter over the image, the stride defines the number of steps to take. Minimum stride is 1.

<div style="text-align:center">
   Stride=1 <img src="/images/Stride1.png" width="500"><br>
   Stride=2 <img src="/images/Stride2.png" width="500">
</div>
<p style="text-align: center;">Effect of a stride of 1 or 2 on the output volume, from [16].</p>

* Padding: Without padding, a convolving filter cannot be centered on the borders of an input, as there will be several missing datapoints. As a result, dimensionality decreases (see Stride example). Padding allows one to define default values (typically 0's, known as zero-padding) for the datapoints outside the image. 

<div style="text-align:center">
    <img src="/images/Pad.png" width="750">
</div>
<p style="text-align: center;">Example of zero-padding of size 2, from [16].</p>

A stride of 1 and a zero-padding fitting the kernel size (e.g., $\frac{KernelSize-1)}{2}$ for odd kernel sizes) produces outputs of equal size to the input. The formula for computing the output dimensionality of a convolutional layer is:

$$OutputSize = \frac{InputSize-KernelSize+2*Padding}{Stride}+1$$

<a name='cnn_volumes'></a>
#### Convolutional Volumes

CNNs are appropriate for any 2-dimensional type of input, such as images. In the case of color images, each input has three channels (RGB) of equal width and height. When processing the input, all three channels should be considered at the same time by any conv filter processing the input (if a filter is processing the pixel located at [x,y], the three RGB values of the pixel should be part of the input). For that purpose, convolutional filters always use the full depth of the input. In the case of the first layer of convolutional filters are 3-dimensional where the width and height are defined by the kernel size, and the depth is 3. 

A single conv filter convolved over the whole input produces a 2-dimensional numerical output. Such 2-dimensional output corresponds to the application of a single filter to the input volume, in a sense it represents the flat (as in 2D) interpretation of the 3D input according to the filter. For every filter we have in a conv layer (i.e., for every convolutional neuron), a different 2-dimensional output is produced. As a result, a convolutional layer produces a 3-dimensional volume, where each 2D slice of the volume corresponds to the output of a single conv neuron. The width and height of the 3D output is defined by the previously defined formula, while the depth is defined by the number of neurons in the layer.

<div style="text-align:center">
    <img src="/images/Figure_5.png" width="700">
</div>
<p style="text-align: center;">Example of 3-dimensional kernel and how this translates into volumes, from [17].</p>

Convolutional filters are always applied in full depth. While the first layer of convolutional neurons processes a 3D input where the depth represents RGB channels, a posterior convolutional layer will process a 3D input where the depth represents the channels defined by the neurons of the first layer. Consequently, the first layer of conv features percieve the input based on the three properties defined by the RGB colors, while consequent conv layers percieve the input based on the properties defined by the neurons from the previous layer. 


<div style="text-align:center">
    <img src="/images/cnn_volumes.png" width="650">
</div>
<p style="text-align: center;">Illustration of how CNN volumes and filters interact, from [18].</p>

#### Pooling

In most cases, a certain degree of translational invariance is desirable: Two identical inputs where one is slightly shifted to the right should be processed by the network as if they were the same. For providing that sort of invariance a commonly used technique are pooling layers. A pooling layer combines data that is spatially related into a single output, like conv neurons do. However, instead of learning a filter for that purpose, pooling layers apply a simple mathematical reduction, such as average or max. As a side effect, pooling layers typically reduce the width and height of the input. This is useful for reducing the network complexity, as the depth of the volume is already increased by convolutional layers (e.g., from 3 channels in a RGB image to X channels after a layer with X conv neurons). On the other hand, pooling layers cause a loss in spatial precision, as the exact location of the input that caused an output of the pooling is unknown.
The main parameters to be defined for a pooling layer are the pooling region and the stride. Unlike convolutional neurons, which are applied full-depth, pooling is only applied to a single channel of the volume.

<div style="text-align:center">
    <img src="/images/pooling1.png" width="450">
</div>
<p style="text-align: center;">Effect of applying a max pooling on the data, from [25].</p>

<div style="text-align:center">
    <img src="/images/pooling2.jpeg" width="450">
</div>
<p style="text-align: center;">Effect of applying a max pooling on the volume, from [25].</p>







<a name='conv_arch'></a>
### Convolutional Architectures

With all the previously described components, many different convolutional neural network architecture could be defined. Certain patterns of layers have been shown to perform particularly well, particularly in image related tasks, and have become a sort of standard. The first influential CNN architecture (known as AlexNet) was defined by Krizhevsky, Sutskever, and Hinton, in 2012 [26]. In their work, Krizhevsky et. al. defined a 5-layer network using convolutions, poolings, ReLUs and Dropout for regularization (regularization is detailed later in this document in the [Regularization Methods](#regularization) section). Krizhevsky et. al. applied their CNN to the most challenging image recognition problem at the moment: ImageNet2012. ImageNet2012 contains 1,000 different classes of images, including many similar classes that are hard to discriminate even for humans (e.g., it includes more than 100 breeds of dogs). Using GPUs for training, which speeds up the process significantly, AlexNet outperformed all the alternative methods based on traditional computer vision approaches, improving the best alternative by 11%. These results had a huge impact on the field, and two years later all ImageNet competitors were based on CNNs.

<div style="text-align:center">
    <img src="/images/imagenet.png" width="750">
</div>
<p style="text-align: center;">Results of the ImageNet challenge. By 2014 all competitors used CNNs. By 2015 human level recognition was achieved.</p>

The AlexNet architecture is composed by 5 pairs of conv-pooling layers, with 3 fully-connected layers on top.

<div style="text-align:center">
    <img src="/images/alexnet.png" width="550">
</div>
<p style="text-align: center;">AlexNet architecture, from [26]. The network is split horizontally because it needed to be trained on two different GPUs.</p>

On top of the last fully-connected layer, either a Softmax or a SVM is placed, to perform the final classification. Other relevant CNN that follow this same scheme are the VGG16/19 [36] (by Oxford VGG research group), the Inception (first called GoogLeNet [37], by Google) and the ResNet [38] (by Microsoft).



<a name='regularization'></a>
### Regularization Methods

The learning capacity of a network is defined by its number of weights and its depth. A deep architecture will typically have millions (AlexNet has 60M parameters, and the VGG16 has 138M parameters), and a majority of those come from the fully-connected layers as conv layers have weight sharing which decreases their number. A single fully-connected layer with 4,096 neurons, where its input layer also has 4,096 neurons, will contain 16M parameters on its own (4,096 x 4,096). A model with so many parameters can easily overfit to any training dataset, as it has the capacity to memorize it. To avoid the overfitting problem, several regularization methods have been proposed for deep architectures. These include:

- L1/L2 regularization: This method tries to avoid spiking values of weights by adding a penalization to their squared magnitude.
- Dropout [27]: This method sets to zero random neurons with a certain probability during training, sparsifying the ANN connectivity in practice. Dropout is most useful on fully-connected layers.
- Batch Normalization [39]: This method normalizes each input batch by both mean and variance, typically after fully-connected layers. It serves both as a regularizing and to solve problems related with weight initialization, while also speeding up convergence.
- Layer Normalization [44]: This method is similar to batch normalization, but mean and variance are computed at layer level. This is particularly useful for RNN (where batch normalization is not directly applicable) and fully-connected layers, but it does not work very well for conv layers. It can speed up convergence even faster than batch normalization.

See [24] for further details on regularization methods.



<a name='inside'></a>
### CNN from the Inside

Historically, neural networks have been considered as "black box" models, as it is hard to understand what is happening within a network during or after its training. The sub-symbolic nature of the representations learnt by a neural net (i.e., lots of numerical parameters in the shape of weights between neurons) makes it impossible to provide any sort of high-level interpretation on those representations. To the human eye, those parameters have no symbolic or semantic meaning (hence, sub-symbolic). 

However, since the popularization of CNNs, a significant amount of effort has been put on opening up neural networks, to try to understand what these models are learning and how. One of the main approches to do so is by visualizing CNN activations or filters. CNN filters of the first layers typically look like Gabor filters. These are the basic features a deep CNN learns for characterizing images. Later layers learn features which are combinations of these, such that the deeper we go in the network, the more complex patterns are learnt.

<div style="text-align:center">
    <img src="/images/filt1.png" width="250">
    <img src="/images/filt2.png" width="500">
</div>
<p style="text-align: center;">Top: Visualization of the filters from the first conv layer. Bottom: Visualization of the filters from the second conv layer. From [29].</p>

<div style="text-align:center">
    <img src="/images/features.png" width="650">
</div>
<p style="text-align: center;">Illustration of how features from different layers grow in complexity. From Quora.</p>

More complex visualization techniques have been developed. The Deep Visualization Toolbox implements some of these, and is a useful tool for playing around with these visualizations. [Deep Visualization Toolbox video](https://www.youtube.com/watch?v=AgkfIQ4IGaM)


<a name='apps'></a>
### CNN Applications

The success of CNNs have resulted in a wide variety of applications. These illustrate the power of these models at characterizing data, and their flexibility to compose more complex solutions. Here are a few examples.

#### Image Classification

Image classification is the basic problem CNN were designed to solve. Given a set of classes, the model has to determine which class an input image belongs to. The most popular datasets for that purpose are MNIST (hand-written digit recognition), CIFAR (low resolution images with 10 and 100 classes), and ImageNet (1,000 classes, including similar classes that are hard to discriminate).

<div style="text-align:center">
    <img src="/images/imagenet_cnn.png" width="550">
</div>
<p style="text-align: center;">Example of classification performance of a CNN on ImageNet, from [26].</p>


#### Image Segmentation

Image segmentation is the problem of finding the exact location of a given entity in an image. The output of the model is classification of pixels, where each pixel is associated with a given identified entity. PASCAL VOC is one of the most popular datasets for this task.

<div style="text-align:center">
    <img src="/images/segmentation.png" width="550">
</div>
<p style="text-align: center;">Example of segmentation results, from [30].</p>


#### Style Transfer

The correlation between the neuron activations of the same layer has a relation with the pictoric style of an input image. For transfering the style from one image to another, first the Gram matrix of a layer is computed on the source style image. Then, on the target style image (often refered as content image), the training procedure is modified so that the Gram matrix of the source image is taken into account when updating the weights. By doing so, the content image can be modified to look similar in style to the source style image.


<div style="text-align:center">
    <img src="/images/style.png" width="550">
</div>
<p style="text-align: center;">Example of style transfer, from [31].</p>


#### Automatic Image Colorization

The purpose of image colorization is to, given an input black and white image, produce a colorization of the same image which is coherent with the contents of it. For that purpose a CNN is connected with a Deconvolutional net, which is essentially a mirrored version of a CNN. This architecture is also used for image segmentation, as it allows classification at pixel-level.

<div style="text-align:center">
    <img src="/images/conv_deconv.png" width="650">
</div>
<p style="text-align: center;">Example of a convolutional and deconvolutional architecture, from [34].</p>
 
<div style="text-align:center">
    <img src="/images/color.png" width="550">
</div>
<p style="text-align: center;">Example of image colorization, from [32].</p>


<div style="text-align:center">
    <img src="/images/color2.png" width="550">
</div>
<p style="text-align: center;">Example of image colorization, from [33].</p>


#### Image/Caption Retrieval

Combining a CNN with a model for representing text, allows one to generate a shared embedding space of both modalities. A set of multimodal pipelines have been proposed with that idea in mind, and have been applied at the task of image captioning (given an image find a textual description for it) and image retrieval (given a textual caption, find a coherent image for it).

<div style="text-align:center">
    <img src="/images/caption.png" width="650">
</div>
<p style="text-align: center;">Example of caption retrieval, from [35].</p>

#### Other applications

There are many other applications of CNNs. In combination with reinforcement learning, CNNs have been trained for playing ATARI videogames or the GO boardgame. CNNs are also a key enabling technology for self-driving cars.



<a name='bib'></a>
## Bibliography

[1] [McCulloch, Warren S., and Walter Pitts. "A logical calculus of the ideas immanent in nervous activity." The bulletin of mathematical biophysics 5.4 (1943): 115-133.](http://vordenker.de/ggphilosophy/mcculloch_a-logical-calculus.pdf)

[2] [Rosenblatt, Frank. "The perceptron: A probabilistic model for information storage and organization in the brain." Psychological review 65.6 (1958): 386](http://www-public.tem-tsp.eu/~gibson/Teaching/Teaching-ReadingMaterial/Rosenblatt58.pdf)

[3] [Mark I Perceptron Operators' Manual](http://www.dtic.mil/dtic/tr/fulltext/u2/236965.pdf)

[4] Kurzweil, Ray. How to create a mind: The secret of human thought revealed. Penguin, 2013.

[5] Misky, M., and S. Papert. "Perceptrons: an introduction to computational geometry." (1969).

[6] [https://en.wikipedia.org/wiki/Perceptrons_(book)](https://en.wikipedia.org/wiki/Perceptrons_(book))

[7] [Werbos, Paul John. "Beyond regression: New tools for prediction and analysis in the behavioral sciences." Doctoral Dissertation, Applied Mathematics, Harvard University (1974)]()

[8] [Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. Learning internal representations by error propagation. No. ICS-8506. California Univ San Diego La Jolla Inst for Cognitive Science, 1985.](http://www.dtic.mil/get-tr-doc/pdf?AD=ADA164453)

[9] [http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/](http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/)

[10] [http://neuralnetworksanddeeplearning.com/chap2.html](http://neuralnetworksanddeeplearning.com/chap2.html)

[11] [LeCun, Yann, et al. "Backpropagation applied to handwritten zip code recognition." Neural computation 1.4 (1989): 541-551.](http://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf)

[12] [Fukushima, Kunihiko. "Neocognitron: A hierarchical neural network capable of visual pattern recognition." Neural networks 1.2 (1988): 119-130.](https://pdfs.semanticscholar.org/c85e/6878f2048d0ec9d7186e3f20592c543635dd.pdf)

[13] [http://intellabs.github.io/RiverTrail/tutorial/](http://intellabs.github.io/RiverTrail/tutorial/)

[14] [http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)

[15] [http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

[16] [https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

[17] [http://xrds.acm.org/blog/2016/06/convolutional-neural-networks-cnns-illustrated-explanation/](http://xrds.acm.org/blog/2016/06/convolutional-neural-networks-cnns-illustrated-explanation/)

[18] [https://www.quora.com/How-is-a-convolutional-neural-network-able-to-learn-invariant-features](https://www.quora.com/How-is-a-convolutional-neural-network-able-to-learn-invariant-features)

[19] [https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

[20] [https://www.quora.com/What-is-the-vanishing-gradient-problem](https://www.quora.com/What-is-the-vanishing-gradient-problem)

[21] [http://cs231n.github.io/neural-networks-1/#actfun](http://cs231n.github.io/neural-networks-1/#actfun)

[22] [http://cs231n.github.io/neural-networks-3/](http://cs231n.github.io/neural-networks-3/)

[23] [https://github.com/fchollet/keras/issues/68](https://github.com/fchollet/keras/issues/68)

[24] [hthttps://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdftp://cs231n.github.io/neural-networks-2/](http://cs231n.github.io/neural-networks-2/)

[25] [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/)

[26] [Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[27] [Srivastava Nitish, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. "Dropout: a simple way to prevent neural networks from overfitting." Journal of Machine Learning Research 15.1 (2014): 1929-1958.](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

[28] [Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).http://cs231n.github.io/understanding-cnn/](http://cs231n.github.io/understanding-cnn/)

[29] [Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolu    tional networks." European conference on computer vision. Springer, Cham, 201    4.](https://arxiv.org/pdf/1311.2901v3.pdf)

[30] [Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." arXiv preprint arXiv:1606.00915 (2016).](https://arxiv.org/pdf/1606.00915.pdf)

[31] [Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).](https://arxiv.org/pdf/1508.06576)

[32] [Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European Conference on Computer Vision. Springer International Publishing, 2016.](https://arxiv.org/pdf/1603.08511)

[33] [Iizuka, Satoshi, Edgar Simo-Serra, and Hiroshi Ishikawa. "Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification." ACM Transactions on Graphics (TOG) 35.4 (2016): 110.](https://pdfs.semanticscholar.org/5c6a/0a8d993edf86846ac7c6be335fba244a59f8.pdf)

[34] [http://www.tensorflowexamples.com/2017/01/transposed-convnets-or-deconvolution.html](http://www.tensorflowexamples.com/2017/01/transposed-convnets-or-deconvolution.html)

[35] [Kiros, Ryan, Ruslan Salakhutdinov, and Richard S. Zemel. "Unifying visual-semantic embeddings with multimodal neural language models." arXiv preprint arXiv:1411.2539 (2014).](https://arxiv.org/pdf/1411.2539)

[36] [Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).](https://arxiv.org/pdf/1409.1556/)

[37] [Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)

[38] [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

[39] [Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International Conference on Machine Learning. 2015.](http://proceedings.mlr.press/v37/ioffe15.html)

[40] [https://www.linkedin.com/pulse/gradient-descent-simple-words-parth-jha](https://www.linkedin.com/pulse/gradient-descent-simple-words-parth-jha)

[41] [http://ruder.io/optimizing-gradient-descent/](http://ruder.io/optimizing-gradient-descent/)

[42] [https://becominghuman.ai/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c](https://becominghuman.ai/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c)

[43] [https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

[44] [Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote)

### Other uncited sources:

[https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/](https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/)

[http://iamaaditya.github.io/2016/03/one-by-one-convolution/](http://iamaaditya.github.io/2016/03/one-by-one-convolution/)

[http://www.robots.ox.ac.uk/~vgg/practicals/cnn/](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/)
