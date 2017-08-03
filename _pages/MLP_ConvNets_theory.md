---
layout: default
mathjax: true
permalink: /mlp-convnets-theory/
---

## Multilayer Perceptron and Convolutional Neural Networks - Theory

This document contains the first theoretical part (1/4) of the Deep Learning subject at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya. It briefly reviews the basic concepts regarding Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNNs).

Table of Contents:

- [A bit of history](#history)
- [Rosenblatt's Perceptron](#rosenblatt)
- [Backpropagation and Stochastic Gradient Descent](#backprop)
- [Convolutional Neural Networks](#cnn)
    - [CNN Parameters](#cnn_params)
    - [CNN Volumes](#cnn_volumes)
- [Activation Functions](#activations)
- [Bibliography](#bib)



<a name='history'></a>
### A bit of history

Neural networks have recently (i.e., 2010s) become a hot topic for both academia and industry. However, neural networks have been around for a while.

Artificial Neural Networks (ANN) were born in 1943, through a work by Warren McCulloch and Walter Pitts [1]. In their paper, McCulloch and Pitts tried to understand how could the brain compute highly complex behaviors, using processing units as simple as neurons. Their design of single neurons was the first contribution to the field, and included the idea of weighted inputs to produce an output.

<div style="text-align:center">
    <img src="/figures/mcculloch_fig1.jpg" width="450">
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
    <img src="/figures/Mark1.png" width="450">
</div>
<p style="text-align: center;">Illustration of the Mark I Perceptron from [3].</p>

Rosenblatt acknolwedged a set of limitations of his Perceptron machine in a series of publications. Minsky and Papert published the book "Perceptrons: an introduction to computational geometry", also detailing some of those limitations. See [6] for more details. The work of Minsky and Papert had a huge impact on the public, although few people actually understood the nature of their contribution. Simply put, Minsky and Papert argue that a multilayer network (or MLP) is needed for learning certain basic functions such as XOR, and that no appropriate training algorithm existed for that kind of network.

Regardless of the technical aspects of Misnky and Papert's work, the reaction from the public was a drastic cut in funding on ANN during the 70s, until the mid 80s, in what is knows as the "AI Winter". After ANNs were almost abandoned, AI research focused on "expert systems" instead, which would also suffer their own "AI Winter" in the 90s.

<a name='backprop'></a>
### Backpropagation and Stochastic Gradient Descent

The backpropagation algorithm reignited the interest on ANN. Originally intended for ANN by Webos in 1974 [7], it gained attention when rediscovered by Rumelhart, Hinton and Williams in 1985 [8], and effectively finished the "AI Winter" on ANN.

Simply put, the backpropagation algorithm is based on the chain rule, which allows to compute the derivative of previous layers, and thus addapt the weights of those layers as well. By doing this process iteratively, we can eventually optimize the weights of all layers in a MLP. In most cases, the change in the weights was computed using an optimization algorithm called Stochastic Gradient Descent (SGD). See [10] for a full mathematical explanation of the backprop algorithm.

With a new training methodology, research on ANN became active again. LeCun et. al. [11] developed a digit recognition system using data from the US Postal Service, and showed how ANN could be used to solve complex practical problems. LeCun system included a layer of convolutional neurons, which had been previously proposed by Fukushima in 1980 [12].

<a name='cnn'></a>
### Convolutional Neural Networks

Convolutional Neural Networks (CNN) are based on a special type of neuron: convolutional neurons. Typically, convolutional neurons have a limited input, e.g., are only connected with a few neurons from the previous layer. Fully-connected neurons on the other hand are connected to all the neurons from the previous layer.

Conv neurons assume the existence of a two-dimensional structure in the data, and are connected to neurons that define squared patches of the previous layer.

<div style="text-align:center">
    <img src="/figures/convolution2.png" width="350">
</div>
<p style="text-align: center;">Illustration of a conv neuron, from [13]. The Kernel in the figure corresponds to the learnt weights of the red neuron. This neuron has 9 inputs (3x3) from the previous layer.</p>

By doing this limited connectivity, conv layers can focus on a particular patch of the input. Significantly, the kernel learnt for that patch may be relevant for all the other patches in the input. Hence, we can define similar neurons that use the same kernel (i.e., weights) but which focus on other parts of the input. This idea is commonly known as "weight sharing", as several neurons of the same layer are defined by a common set of weights, and allows for the use of lots of neurons using the same set of trainable parameters. 

The convolution weight sharing process considers the definition of several neurons with equal weights processing different parts of the input. A different way of understanding this process is to consider that a single filter is slided (or convolved) over all the input, to produce a set of outputs. See the next figure for a visual example.

<div style="text-align:center">
    <img src="/figures/Convolution_schematic.gif" width="350">
</div>
<p style="text-align: center;">Illustration of a filter being convolved, from [14].</p>

Convolving filters can have many effects. Averaging each pixel by its neighbors blurs the image, while taking the difference between the central pixel and the rest can be used to detect edges.


<div style="text-align:center">
   <img src="/figures/convolution-blur.png" width="350"><img src="/figures/generic-taj-convmatrix-blur.jpg" width="350"><br>
   <img src="/figures/convolution-edge-detect1.png" width="350"><img src="/figures/generic-taj-convmatrix-edge-detect.jpg" width="350">
</div>
<p style="text-align: center;">Top: convolved filter to blur image. Bottom: Convolved filter to detect edges. From [15].</p>

<a name='cnn_params'></a>
#### CNN Parameters

The convolution process has many adjustable parameters. The kernel size is the main one, and determines the patch of the image each filter will be able to process. The two other main parameters are the stride and the padding.

* Stride: When applying the same convolving filter over the image, the stride defines the number of steps to take. Minimum stride is 1.

<div style="text-align:center">
   Stride=1 <img src="/figures/Stride1.png" width="500"><br>
   Stride=2 <img src="/figures/Stride2.png" width="500">
</div>
<p style="text-align: center;">Effect of a stride of 1 or 2 on the output volume, from [16].</p>

* Padding: Without padding, a convolving filter cannot be centered on the borders of an input, as there will be several missing datapoints. As a result, dimensionality decreases (see Stride example). Padding allows one to define default values (typically 0's a.k.a. zero-padding) for the datapoints outside the image. A stride of 1 and a zero-padding fitting the kernel size (e.g., $\frac{KernelSize-1)}{2}$ for odd kernel sizes) produces outputs of equal size to the input.

<div style="text-align:center">
    <img src="/figures/Pad.png" width="550">
</div>
<p style="text-align: center;">Example of zero-padding of 2, from [16].</p>

The formula for computing the output of a convolutional layer is:

$$OutputSize = \frac{InputSize-KernelSize+2*Padding}{Stride}+1$$

<a name='cnn_volumes'></a>
#### Convolutional Volumes

Although CNN are appropriate for any 2-dimensional type of input, in most cases these are applied to images. In the case of color images, each data input has three channels (RGB) of equal width and height. When processing the input, all three channels should be considered at the same time by any conv filter processing the input. For that purpose, the first layer of convolutional filters are 3-dimensional where the width and height are defined by the kernel size, and the depth is 3. 

A single conv filter convolved over the whole input (or what is the same, a set of conv neurons sharing the same weights) produces a 2-dimensional numerical output. Such 2-dimensional output corresponds to the application of a single filter to the input volume. For every filter we have in a conv layer (i.e., for every neuron), a new 2-dimensional output is produced. As a result, a convolutional layer produces a 3-dimensional volume, where the width and height is defined by the previously defined formula, and the depth is defined by the number of neurons.

<div style="text-align:center">
    <img src="/figures/Figure_5.png" width="550">
</div>
<p style="text-align: center;">Example of 3-dimensional kernel and how this translates into volumes, from [17].</p>

Consider now this recently created output as the input of a consequent convolution layer. In this case, the conv filters will also be applied in depth, but instead of processing 3 RGB channels, these neurons will be processing as many channels as neurons had the previous layer. The analogy with the color is clear. The first layer of conv features percieve the input in the context of three properties defined by the RGB colors.  Consequent conv layers percieve the input in the context of the properties defined by the previous neurons. 


<div style="text-align:center">
    <img src="/figures/cnn_volumes.png" width="550">
</div>
<p style="text-align: center;">Illustration of how CNN volumes and filters interact, from [18].</p>


<a name='activations'></a>
### Activation Functions

The biologically inspired ANN require an activation function for every neuron, which determines the neuron's output given the weighted sum of its inputs. Rosenblatt's Perceptron used the simplest one, a binary function that either activates with the value, or it does not. Nowadays more complex functions are used, all of them with a positive slope of some sort. The activation function needs to be non-linear, to provide the ANN with the capacity to learn non-linear patterns. 

These include:

The Sigmoid: $f(x)=\frac{1}{1+e^{-x}}$
<div style="text-align:center">
    <img src="/figures/sigmoid.png" width="350">
</div>
<p style="text-align: center;">Sigmoid function, from [19]. See [21] for more details on activation functions.</p>

 The Rectified Linear Unit (ReLU): $f(x)=max(0,x)$
<div style="text-align:center">
    <img src="/figures/relu.jpeg" width="350">
</div>
<p style="text-align: center;">ReLU function, from [19]. See [21] for more details on activation functions.</p>

ReLU are currently being used by default, as these are more efficient. ReLU also avoids the vanishing gradient problem through its constant slope (see [20] for more details on that).


### Convolutional Architectures



### Regularization Methods


### Training Parameters

There are many training paremeters that can be tuned, and finding a good set is not straight-forward. In most cases, its just a matter of try and error, which typically requires executing the training procedure hundreds of times. Next we review some of those.

#### Epochs and batch size

An epoch correponds to a training stage where all images in the training set are used for training once. Typically, an ANN training will require lots of epochs, as each example takes the network parameters in a different direction, and the network cannot learn all there is to learn from an image in a single sight. However when using too many epochs the network will eventually overfit.

The batch size defines the number of images that are processed together in a forward and backward pass of the network. Batch size can go from 1 to full dataset size. Larger batch sizes will train faster, but may be less precise [23]. The most commonly used batch sizes are 16 and 32.

Batch sizes too small may be identified by a noisy function of loss over time. With a batch size equal to the dataset size, variation should be minimal (unless the learning rate is too high).

<div style="text-align:center">
    <img src="/figures/loss.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration of a noisy loss function, from [22].</p>

#### Learning Rate

Learning rate defines the amount of the total derivative of the loss that is applied on each step. Typically, applying the full derivative provides an overcorrection, and smaller steps are required. A learning rate too large will cause the neurons to make large "jumps" within the high-dimensional space being defined, and will prevent them to find global minimums. A learning rate too small may left the neuron stuck in a local minimum, or may make training unfeasibly long. Regardless, its often better to start with a small learning rate and increase it on later experiments.

Plotting the behavior of the loss over time gives a good insight on our choice of learning rate.

<div style="text-align:center">
    <img src="/figures/learningrates.jpeg" width="450">
</div>
<p style="text-align: center;">Illustration on the behavior of various learning rates, from [22].</p>

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
