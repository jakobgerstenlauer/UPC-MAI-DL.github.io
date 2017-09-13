---
permalink: /rnn-lab-guided/
---

## Recurrent Neural Networks (Autonomous Laboratory)

This page contains the guided laboratory of the RNN topic for the Deep Learning course at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya.

You can download the code and data for this examples from the following
github repository [https://github.com/bejar/DLMAI](https://github.com/bejar/DLMAI)


<details>
<summary> Task 1: Time Series Regression (Wind Speed Prediction)</summary>
<p>
The goal of this example is to predict the wind speed of a geographical site
given a window of the previous measurements.

The data for this example has been extracted from the NREL
[Integration National Dataset Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)

This dataset includes metereological information for more than 125,000 sites
across the USA for wind turbine generation prediction.

The dataset included in this example has data for 4 sites and includes the
variable **wind speed at 100m**, **air density**, **temperature** and
**air pressure**.

For this example we are going to use only the **wind speed** variable for one
site. You will use the rest of the data during the autonomous laboratory.

The data and the code are in the `\Wind` directory. There file `Wind.npz`
contains the data matrices for all four sites. The data is in npz numpy format,
this means that of you load the data from the file you will have an object that
stores all the data matrices. This object has the attribute `file` that tells
you the name of the matrices. We are going to use the matix `wind90-45142`.

The code of the example is in the `WindPrediction.py` file.
</p>
</summary>

### Time Series Classification (Electric Devices classification )

The goal of this example is to classify a set of time series corresponding to
the daily power consumption of household devices to one of seven categories.

The data has been _borrowed_ from the
[UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/)
(ElectricDevices dataset)

Each example has 96 attributes corresponding to a measure of the power consumption
every 15 minutes of a whole day. The classes correspond to


The data and the code are in the `\Electric` directory. There are two datafiles:

* `ElectricDevices_TRAIN.csv`
* `ElectricDevices_TEST.csv`

The code of the example is in the `ElectricClass.py` file.

The architecture for this task is composed by:

* A first RNN layer with input size the length of the training window with an
 attribute per element in the sequence.
* Optionally several stacked RNN layers
* A dense layer with softmax activation of size the number of classes

Elements to play with:

* The type of RNN (LSTM, GRU, SimpleRNN)
* The dropout
* The number of layers


### Sequence Classification (Twitter sentiment analysis)

The goal of this example is to classifiy the sentiment of a tweet according to
if it is positive, negative or neutral.

The data and the code are in the `\Sentiment` directory.

There are two datasets `Airlines.csv` and 'Presidential.csv'. The first one
contains tweets about US airlines and the second are tweets about the
2016 US Republican party presidential debate. Both have three classes
(positive, negative and neutral)

The code is in `Sentiment.py`. This code reads the tweets file and generates
a vocabulary with all the different words that appear, everything that does not
correspond with letters a-z is discarded.

The size of the vocabulary can be controled with the `numwords` variable. Words
are recoded as numbers and the tweets are recoded accordingly maintaining the
order of the words in the


The architecture for this task is composed by:

* A first `Embedding` layer that transforms the vectors corresponding to the
tweets to an embedded space
* At least one RNN layer
* A dense layer with softmax activation of size three (the number of classes)

Elements to play with:

* The number of words in the vocabulary of the tweets
* The size of the embedding
* The type of RNN (LSTM, GRU, SimpleRNN)
* The dropout
* The number of layers


### Sequence Classification (Text Generation)

This example has been _borrowed_ from the Keras examples.

The main idea of this example is that we can generate text basically by
predicting for a sequence of letters the next most probable letter.

This is a simplification of a more computationally expensive problem that would
be to predict the next word of a sequence of words. The good thing about this
approach is that the number of classes to predict is just the number of characters
in the text instead of a huge vocabulary dataset.

The data used for the example corresponds to poetry selected from different
English authors. The main reason for choosing poetry instead of narrative is
mainly that poetry has a more relaxed grammar, sentences are shorter and usually
the main topic of the text is expressed in a limited number of words.

For this example four datasets of different length have been generated
(poetry1.txt to poetry4.txt). Each dataset is included in the next one.

The data and the code are in the `\TextGeneration` directory.

The data is transformed to one-hot encoding for building the classifier, assuming
that the number of classes is the number of unique characters in the text.

The code is in `TextGenerator.py`. This code reads the data file and generates
a training set by sequentially extracting windows of size `maxlen` moving
the window `step` number of characters every iteration. For each window the class
correspond to the next character. The classes are transformed to one-hot encoding
of the size the number of different characters in the text.


The architecture for this task is composed by:

* A first RNN layer with input size the length of the training window with as
 many attributes per element in the sequence as unique characters.
* Optionally several stacked RNN layers
* A dense layer with softmax activation of size the number or characters

Elements to play with:

* The type of RNN (LSTM, GRU, SimpleRNN)
* The dropout
* The number of layers


### Sequence to sequence (Addition)

This example has also been _borrowed_ from the Keras examples.

The idea is to show a simple example that learns associations between sequences.

The input sequences correspond to additions that have a maximum length assuming
that the figures added have a limited number of digits, for example `1234+54`.
The output sequences are the correct answer to the addition.

The main advantage of this examples is that it is very easy to generate examples
for the problem and it is easy to coverge to a solution with 99% of accuracy.

The architecture for this problem is the following:
