---
permalink: /embedding-spaces-lab-guided/
---

This page contains the guided laboratory of the Embedding Spaces topic for the Deep Learning course at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya.

Table of Contents:

- [Other sources for experimentation](#other)

---

---


<a name='basic'></a>
## Basic word2vec experiments

Training word2vec requires the processing of huge amounts of text. Although word2vec defines a very shallow network (i.e., it only has one hidden layer), training it takes hours. To experiment with word embeddings, we will use pre-trained embeddings provided for GloVe. These can be downloaded from [here](https://nlp.stanford.edu/projects/glove/). There are several embeddings using different corpus. Download one or all of them.

Since we are not going to train the word embeddings, GPUs are not necessary for these experiments. You can still use Minotauro for executing your jobs, but you can also do it on your personal computer. Take into account that these codes are not parallelized. For efficient use of Minotauro, consider parallelizing the code using standard Python libraries (see the multiprocessing module).

### Loading word embeddings

First, lets load the pre-computed word embeddings into a dictionary. The resultant data structure will contain as key the word, and as value the 1D embedding.

```python
import os
import numpy as np

#Create a dictionary/map to store the word embeddings
embeddings_index = {}

#Load pre-computed word embeddings
#These can be dowloaded from https://nlp.stanford.edu/projects/glove/
#e.g., wget http://nlp.stanford.edu/data/glove.6B.zip
embeddings_size = "300"
f = open(os.path.join('.', 'glove.6B.'+embeddings_size+'d.txt'))

#Process file and load into structure
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

```

### Finding the most similar word

With the embeddings loaded, we can now compute distances. Let's find the most similar word for each word in the dictionary. We will build a distance matrix, of size equal to the vocabulary size. Depending on the machine you run this code, you may need to reduce the number of words to make the distance matrix fit into memory.

```python
#Compute distances among first X words
max_words = 10000
from sklearn.metrics.pairwise import pairwise_distances
mat = pairwise_distances(embeddings_index.values()[:max_words])

#Replace self distances from 0 to inf (to use argmin)
np.fill_diagonal(mat, np.inf)

#Find the most similar word for every word
min_0 = np.argmin(mat,axis=0)

#Save the pairs to a file
f_out = open('similarity_pairs_dim'+embeddings_size+'_first'+str(max_words)+'.txt','w')
for i,item in enumerate(embeddings_index.keys()[:max_words]):
    f_out.write(str(item)+' '+str(embeddings_index.keys()[min_0[i]])+'\n')
```

### Testing the "king - man + woman = queen" analogy

Now let's test the famous analogy "king - man + woman = queen". To do so, we simply have to operate arithmetically with the vectors, and find the word vector that is most similar to the result.

```python
#Compute embedding of the analogy
embedding_analogy = embeddings_index['king'] - embeddings_index['man'] + embeddings_index['woman']
#Find distances with the rest of the words
analogy_distances = np.empty(len(embeddings_index))
for i,item in enumerate(embeddings_index.values()):
    analogy_distances[i] = pairwise_distances(embedding_analogy.reshape(1, -1),item.reshape(1, -1))
#Print top 10 results
print [embeddings_index.keys()[i] for i in analogy_distances.argsort()[:10]]
```

Can you come up with any other analogy?


<a name='other'></a>
## Other sources for experimentation

Beyond the codes explained in class, there are other online resources of interest that may be used for experimentation.

### Word2vec

The original code of word2vec, as released by its authors, can be found [here](https://code.google.com/archive/p/word2vec/).

### Kaggle word2vec tutorial

[Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial) is a tutorial for understanding and operating with the word2vec model.

### McCormick inspect word2vec

This [repository](https://github.com/chrisjmccormick/inspect_word2vec) uses gensim in Python to load word2vec pre-trained model, and inspects some of the details of the vocabulary. 

### word2vec tutorial in TensorFlow

Tutorial of word2vec using TensorFlow: [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)

### Word2vec in Keras

[Using Gensim Word2Vec Embeddings in Keras](http://ben.bolte.cc/blog/2016/gensim.html). A short post and script regarding using Gensim Word2Vec embeddings in Keras, with example code.

[Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html). Official.

[CBOW implementation in Keras without dependencies](https://github.com/abaheti95/Deep-Learning/blob/master/word2vec/keras/cbow_model.py)

[StackOverflow details](https://stackoverflow.com/questions/40244101/implement-word2vec-in-keras)

### Gensim

[Official gensim tutorials](https://radimrehurek.com/gensim/tutorial.html)
