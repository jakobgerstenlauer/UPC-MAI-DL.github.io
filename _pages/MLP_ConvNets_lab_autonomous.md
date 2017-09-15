---
permalink: /mlp-convnets-lab-autonomous/
---

The autonomous laboratory is open for your exploration. Using the basic code provided in the guided laboratory, students should modify it to test different settings, and understand the effect of the various components. You could use any dataset you wish. The list of datasets available at Keras can be found [here](https://keras.io/datasets/), but using one of those is not a requirement.

In this document there are a list of suggested questions you may consider to explore. It is not necessary to answer any or all these questions. The student may choose to perform different experiments, for explore different aspects of neural networks. Go ahead and play.

<a name='basic_nn'></a>
### Example 1: Basic NN

- What is the impact of using more fully connected layers?
- What is the impact of increasing the number of neurons per layer?
- What is the best performance you can get out of a basic neural net?



<a name='cnn'></a>
### CNN example

- Can you design and train a model that overfits on the training data (or on a subset of it)?
- When overfitting, what is the result of applying various regularization techniques?
- When using ReLUs, how many neurons are dead after a training?
- Adding data augmentation improves performance?
- How do the different learning algorithms behave for equal architectures? Does regularization have the same affect when using different algorithms?
- How hard is it to match the performance of an adaptative algorithm (e.g., Adam) by using an algorithm where parameters have to be hand tunned (e.g., SGD)?
- What is the result of using different weight initializations in the training process?
