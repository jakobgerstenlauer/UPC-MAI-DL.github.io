---
permalink: /mlp-convnets-theory/
---

## Multilayer Perceptron and Convolutional Neural Networks - Theory

This page contains the first theoretical part (1/4) of the Deep Learning subject at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya. It briefly reviews the basic concepts regarding Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNNs).

Table of Contents:

- [A Bit of History](#history)
- [Rosenblatt's Perceptron](#rosenblatt)
- [Backpropagation and Stochastic Gradient Descent](#backprop)
- [Convolutional Neural Networks](#cnn)
    - [CNN Parameters](#cnn_params)
    - [CNN Volumes](#cnn_volumes)
- [Activation Functions](#activations)
- [CNN Architectures](#conv_arch)
- [Regularization Methods](#regularization)
- [Training Parameters](#training_params)
    - [Epochs and Batch Size](#epochs_batch)
    - [Learning Rate](#learning_rate)
    - [Other Parameters](#other_params)
    - [Adaptative Learning Methods](#adaptative_methods)
- [CNN from the Inside](#inside)
- [CNN Applications](#apps)
- [Bibliography](#bib)



<a name='history'></a>
### A Bit of History

Neural networks have recently (i.e., 2010s) become a hot topic for both academia and industry. However, neural networks have been around for a while.

Artificial Neural Networks (ANN) were born in 1943, through a work by Warren McCulloch and Walter Pitts [1]. In their paper, McCulloch and Pitts tried to understand how could the brain compute highly complex behaviors, using processing units as simple as neurons. Their design of single neurons was the first contribution to the field, and included the idea of weighted inputs to produce an output.

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

Rosenblatt acknolwedged a set of limitations of his Perceptron machine in a series of publications. Minsky and Papert published the book "Perceptrons: an introduction to computational geometry", also detailing some of those limitations. See [6] for more details. 
The work of Minsky and Papert had a huge impact on the public, although few people actually understood the nature of their contribution. Simply put, Minsky and Papert argue that a multilayer network (or MLP) is needed for learning certain basic functions such as XOR, and that no appropriate training algorithm existed for that kind of network.

Regardless of the technical aspects of Misnky and Papert's work, the reaction from the public was a drastic cut in funding on ANN during the 70s, until the mid 80s, in what is knows as the "AI Winter". After ANNs were almost abandoned, AI research focused on "expert systems" instead, which would also suffer their own "AI Winter" in the 90s.

<a name='backprop'></a>
### Backpropagation and Stochastic Gradient Descent

The backpropagation algorithm reignited the interest on ANN. Originally intended for ANN by Webos in 1974 [7], it gained attention when rediscovered by Rumelhart, Hinton and Williams in 1985 [8], and effectively finished the "AI Winter" on ANN.

The backpropagation starts with a forward pass of an input. After the input has reached the last layer of the network, the predicted label (the network output) is compared against the ground truth label, and the error made is measured using a loss function. The computed loss is used to update the weights of the network, finding the best gradient towards minimizing the error (though a partial derivative). To train a neural net with several layers, we need to update the weights from the previous layers as well. This is accomplished by using the chain rule, which allows to compute the derivative of previous layers, and thus update the weights of those layers as well. Traditionally, the change in the weights has been computed using an optimization algorithm called Stochastic Gradient Descent (SGD). SGD iteratively estimates the best direction for optimization using a subset of the whole dataset (hence, stochastic). Since then, several alternatives have been proposed.   

By doing this process iteratively (forward pass, loss function, backpropagate error, weight update), we can eventually optimize the weights of all layers in a MLP. See [10] for a full mathematical explanation of the backprop algorithm. With a new training methodology, research on ANN became active again. LeCun et. al. [11] developed a digit recognition system using data from the US Postal Service, and showed how ANN could be used to solve complex practical problems. LeCun system included a layer of convolutional neurons, which had been previously proposed by Fukushima in 1980 [12].

<a name='cnn'></a>
### Convolutional Neural Networks

Convolutional Neural Networks (CNN) are based on a special type of neuron: convolutional neurons. Typically, convolutional neurons have a limited input, e.g., are only connected with a few neurons from the previous layer. Fully-connected neurons on the other hand are connected to all the neurons from the previous layer.

Conv neurons assume the existence of a two-dimensional structure in the data, and are connected to neurons that define squared patches of the previous layer.

<div style="text-align:center">
    <img src="/images/convolution2.png" width="350">
</div>
<p style="text-align: center;">Illustration of a conv neuron, from [13]. The Kernel in the figure corresponds to the learnt weights of the red neuron. This neuron has 9 inputs (3x3) from the previous layer.</p>

By doing this limited connectivity, conv layers can focus on a particular patch of the input. Significantly, the kernel learnt for that patch may be relevant for all the other patches in the input. Hence, we can define similar neurons that use the same kernel (i.e., weights) but which focus on other parts of the input. This idea is commonly known as "weight sharing", as several neurons of the same layer are defined by a common set of weights, and allows for the use of lots of neurons using the same set of trainable parameters. 

The convolution weight sharing process considers the definition of several neurons with equal weights processing different parts of the input. A different way of understanding this process is to consider that a single filter is slided (or convolved) over all the input, to produce a set of outputs. See the next figure for a visual example.

<div style="text-align:center">
    <img src="/images/Convolution_schematic.gif" width="350">
</div>
<p style="text-align: center;">Illustration of a filter being convolved, from [14].</p>

Convolving filters can have many effects. Averaging each pixel by its neighbors blurs the image, while taking the difference between the central pixel and the rest can be used to detect edges.


<div style="text-align:center">
   <img src="/images/convolution-blur.png" width="200"><img src="/images/generic-taj-convmatrix-blur.jpg" width="300"><br>
   <img src="/images/convolution-edge-detect1.png" width="200"><img src="/images/generic-taj-convmatrix-edge-detect.jpg" width="300">
</div>
<p style="text-align: center;">Top: Convolved filter to blur image. Bottom: Convolved filter to detect edges. From [15].</p>

<a name='cnn_params'></a>
#### CNN Parameters

The convolution process has many adjustable parameters. The kernel size is the main one, and determines the patch of the image each filter will be able to process. The two other main parameters are the stride and the padding.

* Stride: When applying the same convolving filter over the image, the stride defines the number of steps to take. Minimum stride is 1.

<div style="text-align:center">
   Stride=1 <img src="/images/Stride1.png" width="500"><br>
   Stride=2 <img src="/images/Stride2.png" width="500">
</div>
<p style="text-align: center;">Effect of a stride of 1 or 2 on the output volume, from [16].</p>

* Padding: Without padding, a convolving filter cannot be centered on the borders of an input, as there will be several missing datapoints. As a result, dimensionality decreases (see Stride example). Padding allows one to define default values (typically 0's a.k.a. zero-padding) for the datapoints outside the image. A stride of 1 and a zero-padding fitting the kernel size (e.g., $\frac{KernelSize-1)}{2}$ for odd kernel sizes) produces outputs of equal size to the input.

<div style="text-align:center">
    <img src="/images/Pad.png" width="750">
</div>
<p style="text-align: center;">Example of zero-padding of 2, from [16].</p>

The formula for computing the output of a convolutional layer is:

$$OutputSize = \frac{InputSize-KernelSize+2*Padding}{Stride}+1$$

<a name='cnn_volumes'></a>
#### Convolutional Volumes

Although CNN are appropriate for any 2-dimensional type of input, in most cases these are applied to images. In the case of color images, each data input has three channels (RGB) of equal width and height. When processing the input, all three channels should be considered at the same time by any conv filter processing the input. For that purpose, the first layer of convolutional filters are 3-dimensional where the width and height are defined by the kernel size, and the depth is 3. 

A single conv filter convolved over the whole input (or what is the same, a set of conv neurons sharing the same weights) produces a 2-dimensional numerical output. Such 2-dimensional output corresponds to the application of a single filter to the input volume. For every filter we have in a conv layer (i.e., for every neuron), a new 2-dimensional output is produced. As a result, a convolutional layer produces a 3-dimensional volume, where the width and height is defined by the previously defined formula, and the depth is defined by the number of neurons.

<div style="text-align:center">
    <img src="/images/Figure_5.png" width="700">
</div>
<p style="text-align: center;">Example of 3-dimensional kernel and how this translates into volumes, from [17].</p>

Consider now this recently created output as the input of a consequent convolution layer. In this case, the conv filters will also be applied in depth, but instead of processing 3 RGB channels, these neurons will be processing as many channels as neurons had the previous layer. The analogy with the color is clear. The first layer of conv features percieve the input in the context of three properties defined by the RGB colors.  Consequent conv layers percieve the input in the context of the properties defined by the previous neurons. 


<div style="text-align:center">
    <img src="/images/cnn_volumes.png" width="650">
</div>
<p style="text-align: center;">Illustration of how CNN volumes and filters interact, from [18].</p>

#### Pooling

The convolution process increases significantly the size of the volume from layer to layer, typically because we will want to have at least a few tens of neurons per layer. To avoid the size of the volume to explode layer by layer, pooling layers are often added after each convolutional layer. A pooling layer exploits the same spatial information conv layer does, and aggregates data that is spatially related. The two main types of pooling are max pooling and average pooling, with the first being the most common. The other parameters to be set are the size of the pooling region and the stride.

<div style="text-align:center">
    <img src="/images/pooling1.png" width="450">
</div>
<p style="text-align: center;">Effect of applying a max pooling on the data, from [25].</p>

<div style="text-align:center">
    <img src="/images/pooling2.jpeg" width="450">
</div>
<p style="text-align: center;">Effect of applying a max pooling on the volume, from [25].</p>

Pooling layers cause a loss in spatial precision, as the network does not exactly know the location of the original value. On the other hand, pooling provides a certain degree of translation invariance, as slightly shifted input will be captured equally by the pooling layer.



<a name='activations'></a>
### Activation Functions

The biologically inspired ANN require an activation function for every neuron, which determines the neuron's output given the weighted sum of its inputs. Rosenblatt's Perceptron used the simplest one (Equation 1), a binary function that either activates with the value, or it does not. Nowadays more complex functions are used, all of them with a positive slope of some sort. The activation function needs to be non-linear, to provide the ANN with the capacity to learn non-linear patterns. 

These include:

The Sigmoid: $f(x)=\frac{1}{1+e^{-x}}$
<div style="text-align:center">
    <img src="/images/sigmoid.png" width="350">
</div>
<p style="text-align: center;">Sigmoid function, from [19]. See [21] for more details on activation functions.</p>

 The Rectified Linear Unit (ReLU): $f(x)=max(0,x)$
<div style="text-align:center">
    <img src="/images/relu.jpeg" width="350">
</div>
<p style="text-align: center;">ReLU function, from [19]. See [21] for more details on activation functions.</p>

ReLU are currently being used by default, as these are more efficient. ReLU also avoids the vanishing gradient problem through its constant slope (see [20] for more details on that).


<a name='conv_arch'></a>
### Convolutional Architectures

With all the previously described components, one must define a convolutional neural network architecture. There are certain patterns of layers that have been shown to perform particularly well, and in most cases its better to use one of those. The first popular CNN architecture was defined by Krizhevsky, Sutskever, and Hinton [26]. In their work, Krizhevsky et. al. defined a 5-layer network using convolutions, poolings, ReLUs and Dropout for regularization, and managed to win the ImageNet2012 improving the best alternative by 11%. ImageNet2012 was a image recognition problem containing 1,000, including more than 100 breeds of dogs. The training was made possible through the use of GPUs, which speed up the process significantly. 

<div style="text-align:center">
    <img src="/images/imagenet.png" width="750">
</div>
<p style="text-align: center;">Results of the ImageNet challenge. By 2014 all competitors used CNNs. By 2015 human level recognition was achieved.</p>

The AlexNet architecture follows the pattern of pairs of conv-pooling layers, with a few fully-connected layers on top. This pattern has become a standard, and most CNN architectures currently use it.
<div style="text-align:center">
    <img src="/images/alexnet.png" width="550">
</div>
<p style="text-align: center;">AlexNet architecture, from [26]. The network is split horizontally because it needed to be trained on two different GPUs.</p>

On top of the fully-connected layers, either a Softmax or a SVM is placed, to perform the final classification. Other relevant CNN that follow this same scheme are the VGG16/19 (by Oxford VGG research group), the Inception (first called GoogLeNet, by Google) and the ResNet (by Microsoft).



<a name='regularization'></a>
### Regularization Methods

Deep neural networks have a huge learning capacity, defined by the number of parameters in the network. To avoid overfitting, there a set of regularization methods have been proposed. These include:

- L1/L2 regularization
- Dropout [27]
- Batch Normalization

See [24] for further details.


<a name='training_params'></a>
### Training Parameters

There are many training paremeters that can be tuned, and finding a good set is not straight-forward. In most cases, its just a matter of try and error, which typically requires executing the training procedure hundreds of times. Next we review some of those.


<a name='epochs_batch'></a>
#### Epochs and batch size

An epoch correponds to a training stage where all images in the training set are used for training once. Typically, an ANN training will require lots of epochs, as each example takes the network parameters in a different direction, and the network cannot learn all there is to learn from an image in a single sight. However when using too many epochs the network will eventually overfit.

The batch size defines the number of images that are processed together in a forward and backward pass of the network. Batch size can go from 1 to full dataset size. Larger batch sizes will train faster, but may be less precise [23]. The most commonly used batch sizes are 16 and 32.

Batch sizes too small may be identified by a noisy function of loss over time. With a batch size equal to the dataset size, variation should be minimal (unless the learning rate is too high).

<div style="text-align:center">
    <img src="/images/loss.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration of a noisy loss function, from [22].</p>



<a name='learning_rate'></a>
#### Learning Rate

Learning rate defines the amount of the total derivative of the loss that is applied on each step. Typically, applying the full derivative provides an overcorrection, and smaller steps are required. A learning rate too large will cause the neurons to make large "jumps" within the high-dimensional space being defined, and will prevent them to find global minimums. A learning rate too small may left the neuron stuck in a local minimum, or may make training unfeasibly long. Regardless, its often better to start with a small learning rate and increase it on later experiments.

Plotting the behavior of the loss over time gives a good insight on our choice of learning rate.

<div style="text-align:center">
    <img src="/images/learningrates.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration on the behavior of various learning rates, from [22].</p>



<a name='other_params'></a>
#### Other parameters

There are many other parameters that may affect the learning process. We will not detail them here, but the interested student should read about them. These include:

- Weight Decay
- Weight initialization
- Momentum

[22] contains several tips on how to optimize all these paremeters (Hyperparameter optimization Section).



<a name='adaptative_methods'></a>
#### Adaptative learning methods

The number of learning parameters to be fit has motivated the appearence of learning methods which adapt these parameters automatically. These are used instead of stochastic gradient descent, and sometimes can simplify the parameter tuning process significantly.

- Adagrad
- Adadelta
- RMSprop
- Adam



<a name='inside'></a>
### CNN from the Inside

Historically, neural networks have been considered as "black box" models, as it is hard to understand what is happening within a network during or after its training. The sub-symbolic nature of the representations learnt by a neural net (i.e., lots of numerical parameters in the shape of weights between neurons) makes it impossible to provide any sort of high-level interpretation on those representations. However, since the popularization of CNNs, a significant amount of effort has been put on opening up neural networks. One of the main approches to do so is by visualizing CNN activations or filters.

CNN filters of the first layers typically look like Gabor filters. These are the basic features a deep CNN learns for characterizing images. Later layers learn features which are combinations of these, such that the deeper we go in the network, the more complex patterns are learnt.

<div style="text-align:center">
    <img src="/images/filt1.png" width="250">
    <img src="/images/filt2.png" width="500">
</div>
<p style="text-align: center;">Top: Visualization of the filters from the first conv layer. Bottom: Visualization of the filters from the second conv layer. From [29].</p>

<div style="text-align:center">
    <img src="/images/features.png" width="650">
</div>
<p style="text-align: center;">Illustration of how features from different layers grow in complexity. From Quora.</p>

The Deep Visualization Toolbox is a useful tool for playing around with some of these visualizations. [Deep Visualization Toolbox video] (https://www.youtube.com/watch?v=AgkfIQ4IGaM)


<a name='apps'></a>
### CNN Applications

The success of CNNs have resulted in a wide variety of applications. These illustrate the power of these models at characterizing data, and their flexibility to compose more complex solutions. Here are a few examples.

#### Image Classification

Image classification is the basic problem CNN were designed to solve. Given a set of classes, the model has to determine which class an input image belongs to. The most popular datasets for that purpose are MNIST (hand-written digit recognition), CIFAR (low resolution images with 10 and 100 classes), and ImageNet (1,000 classes, including similar subsets such as more than 100 breeds of dogs).

<div style="text-align:center">
    <img src="/images/imagenet_cnn.png" width="550">
</div>
<p style="text-align: center;">Example of classification performance of a CNN on ImageNet, from [26].</p>


#### Image Segmentation

Image segmentation is the problem of finding the exact location of a given entity in an image. The output of the model is classification of pixels, where each pixel is associated with a given identified entity.PASCAL VOC is one of the most popular datasets for this task.

<div style="text-align:center">
    <img src="/images/segmentation.png" width="550">
</div>
<p style="text-align: center;">Example of segmentation results, from [30].</p>


#### Style Transfer

Recently, it was found that the correlation between the neuron activations of the same layer had a relation with the pictoric style of the image. For transfering the style from one image to another, first the Gram matrix of a layer is computed on the source style image. Then, on the target style image, the training procedure is modified so that the Gram matrix of the source image is taken into account. By doing so, the target image is effectively modified to look similar in style to the source image.


<div style="text-align:center">
    <img src="/images/style.png" width="550">
</div>
<p style="text-align: center;">Example of style transfer, from [31].</p>


#### Automatic Image Colorization

The purpose of image colorization is to, given an input black and white image, produce a colorization of the same image which is coherent with the contents of it. For that purpose a CNN is connected with a Deconvolutional net, which is essentially a mirrored version of a CNN. This is the same approach used for image segmentation.

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

There are many other applications of CNNs. These include:

- In combination with reinforcement learning for playing ATARI videogames
- In combination with reinforcement learning for playing GO
- Enabling self-driving cars
- ...




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

[28] [Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).http://cs231n.github.io/understanding-cnn/] (http://cs231n.github.io/understanding-cnn/)

[29] [Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolu    tional networks." European conference on computer vision. Springer, Cham, 201    4.] (https://arxiv.org/pdf/1311.2901v3.pdf)

[30] [Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." arXiv preprint arXiv:1606.00915 (2016).] (https://arxiv.org/pdf/1606.00915.pdf)

[31] [Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).] (https://arxiv.org/pdf/1508.06576)

[32] [Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European Conference on Computer Vision. Springer International Publishing, 2016.] (https://arxiv.org/pdf/1603.08511)

[33] [Iizuka, Satoshi, Edgar Simo-Serra, and Hiroshi Ishikawa. "Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification." ACM Transactions on Graphics (TOG) 35.4 (2016): 110.] (https://pdfs.semanticscholar.org/5c6a/0a8d993edf86846ac7c6be335fba244a59f8.pdf)

[34] [http://www.tensorflowexamples.com/2017/01/transposed-convnets-or-deconvolution.html] (http://www.tensorflowexamples.com/2017/01/transposed-convnets-or-deconvolution.html)

[35] [Kiros, Ryan, Ruslan Salakhutdinov, and Richard S. Zemel. "Unifying visual-semantic embeddings with multimodal neural language models." arXiv preprint arXiv:1411.2539 (2014).] (https://arxiv.org/pdf/1411.2539)

### Other uncited sources:

[https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/](https://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/)

[http://iamaaditya.github.io/2016/03/one-by-one-convolution/] (http://iamaaditya.github.io/2016/03/one-by-one-convolution/)
