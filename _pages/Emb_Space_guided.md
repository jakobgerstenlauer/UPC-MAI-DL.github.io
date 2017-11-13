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


<a name='image_emb'></a>
## Image embedding experiments

Processing image embeddings of a visual dataset only requires a dataset of images and a pre-trained deep architecture. Fortunately, pre-trained deep models are often publictly available for any deep learning tool. In our case, we are going to use the vgg16 architecture pre-trained using the ImageNet2012 dataset (Surely you didn't expect to be ImageNet dataset!). In order to use the pre-trained wieghts of the vgg16 architecture in Keras, we will have to download the weights file and copy the file into the GPFS system:

```
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
scp vgg16_weights_tf_dim_ordering_tf_kernels.h5 USERNAME@dt01.bsc.es:.keras/models/.
```

We will work with the MIT-67 dataset, an indoor scene recognition task. We are going to convert all image representations to embedding representations encoding the visual patterns based on the language learnt by the model. Before processing images from MIT-67, we will have to download its training subset, copy it into the GPFS system and decompress it:

```
wget http://147.83.200.110:8000/static/ferran/mit67_img_train.tar.gz
scp mit67_img_train.tar.gz USERNAME@dt01.bsc.es:.
ssh USERNAME@mt1.bsc.es
tar -xzvf mit67_img_train.tar.gz
```

### Converting images to image embeddings

Converting images to embeddings from single layer with Keras is straigh forward. We have to load the pre-trained weights into the vgg16 architecture to input a batch of images and obtain activations at the previous to last fully-connected layer:

```python
from keras.applications.vgg16 import VGG16
from keras.models import Model
# Load the vgg16 architecture with pre-trained weights using ImageNet2012 dataset.
base_model = VGG16(weights='imagenet')
# Define a custom model that given an input, outputs activations from requested layer (e.g., fc2).
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
```

Then iteratively convert all images to its embedding representations and create a matrix containing all image embeddings:

```python
import sys, time
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from glob import glob

# Define a dataset iterator by batches of an specific size.
def input_pipeline(image_files, batch_size):
    end_flag = False
    for n in range(len(image_files)):
        x_batch = np.zeros((0,224,224,3))
        y_batch = []
        for i in range(batch_size):
            try:
                img_path = image_files.pop(0)
            except IndexError:
                end_flag = True
                break
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x_batch = np.concatenate((x_batch, x), axis=0)
            y = img_path.split('/')[-2]
            y_batch.append(y)
        if end_flag:
            if batch.shape[0]>0:
                yield x_batch, y_batch
        else:
            yield x_batch, y_batch

# Create a list containing all image_paths. 
image_files = glob('~/mit67_img_train/*/*')
step = 0
tot = len(image_files)/10
if len(image_files)%10 > 0:
    tot += 1
# Batching loop.
for x_batch, y_batch in input_pipeline(image_files, 10):
    t0 = time.time()
    # Preprocessing input images for the vgg16 model.
    x = preprocess_input(x_batch)
    # Obtain the embeddings of current batch of images.
    batch_emb = model.predict(x)
    dataset_emb = np.concatenate((dataset_emb, batch_emb))
    dataset_lab += y_batch

    step += 1
    print 'step: {s}/{tot} in {t}s'.format(s=step, tot=tot, t=time.time()-t0)
    sys.stdout.flush()
```

Finally, save the matrix containing all image embeddings and its corresponding labels for further manipulation of the data:

```python
import numy as np

# Saving image embeddings and labels into a numpy file.
np.savez('mit67_embeddings.npz', embeddings=dataset_emb, labels=dataset_lab)
```

### Visualize image embeddings

Once we have the MIT-67 dataset processesed, we can visualize some classes in the embedding space. Let's visualize the first 5 classes:

```python
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# Load dataset embeddings and labels.
obj = np.load('mit67_embeddings.npz')
labels = obj['labels']
embeddings = obj['embeddings']

# Set up the subset of labels to visualize.
n_labels = 5
_uniq = np.unique(labels)[:n_labels]
print 'The {n} labels selected to visualize:'.format(n=n_labels), _uniq

# Create the embeddings matrix (and its corresponding labels list) 
# containing instances belonging to labels in the previous subset.
subsets_emb = np.zeros((0,4096))
lab_range = {}
offset = 0
for idx, _lab in enumerate(_uniq):
    part_emb = embeddings[np.where(labels == _lab)]
    subsets_emb = np.concatenate((subsets_emb, part_emb))
    lab_range[_lab] = [offset, offset+part_emb.shape[0]]
    offset += part_emb.shape[0]
print 'subsets_emb shape:', subsets_emb.shape

# Apply a dimensionality reduction technique to visualize 2 dimensions.
vis_matrix = TSNE(n_components=2).fit_transform(subsets_emb)

# Using matplotlib to create the plot and show it!
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, n_labels)]
plt.figure()
for idx, _lab in enumerate(_uniq):
    r = lab_range[_lab]
    plt.scatter(vis_matrix[r[0]:r[1],0], vis_matrix[r[0]:r[1],1], s=30, c=colors[idx], label=_lab)
plt.legend(loc='best')
plt.show()
```

In this script, we first load the data previously saved. Then we select the first 5 labels of the MIT-67 dataset. Thirdly, we create the embedding matrix containing only instances belonging to the subset of 5 classes. After that, we apply a dimensionality reduction technique called TSNE to reduce the number of features of the data to 2 features, so we are able to visualize instances. In the final block of code, we use the matplotlib tool to visualize instances. Each instance has been colored based on its correponding label.

How separable do you perceive the data using the first 5 labels? If we use the first 10 labels, would remain that separable? Try it.

### Clustering using the image embeddings

Last but not least, we can perform a clustering over the processed datatset. We are going to use the first 5 labels of the dataset and obtain the embedding matrix containing instances from these labels. Afterwards, we apply the typical clustering technique called KMeans and evaluate its performance using the Normalized Mutual Information metric (NMI).

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.decomposition import PCA

# Load dataset embeddings and labels.
obj = np.load('mit67_embeddings.npz')
labels = obj['labels']
embeddings = obj['embeddings']

# Set up the subset of labels to visualize.
n_labels = 5
_uniq = np.unique(labels)[:n_labels]
print 'The {n} labels selected to visualize:'.format(n=n_labels), _uniq

# Create the embeddings matrix (and its corresponding labels list)
# containing instances belonging to labels in the previous subset.
subsets_emb = np.zeros((0,4096))
true_labels = []
for idx, _lab in enumerate(_uniq):
    part_emb = embeddings[np.where(labels == _lab)]
    subsets_emb = np.concatenate((subsets_emb, part_emb))
    true_labels += [_lab for _ in range(part_emb.shape[0])]
true_labels = np.array(true_labels)

# Use a dimensionality reduction technique (e.g., PCA).
reduced_emb = PCA(n_components=100).fit_transform(subsets_emb)

# Apply a clustering technique (e.g., KMeans) and evaluate its performance with
# a metric (e.g., Normalized Mutual Information).
pred_labels = KMeans(n_clusters=n_labels, random_state=0).fit_predict(reduced_emb)
nmi = NMI(true_labels, pred_labels)
print 'NMI score:', nmi
```

Clustering is completely unsupervised. Algorithm consider that two images should be in the same cluster based on a prior encoded in the algorithm. Are you able to apply a classification on the data instead of a clustering? Use an [SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) model from the sklearn python tool.



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
