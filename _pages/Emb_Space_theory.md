---
permalink: /emb-space-theory/
---

This page contains the theoretical part of the Embedding Spaces topic for the Deep Learning course at the Master in Artificial Intelligence of the Universitat Politècnica de Catalunya. 

Table of Contents:

- [Introduction](#intro)
- [Word Embeddings](#wordemb)
    - [word2vec](#word2vec)
        - [skipgram](#skipgram)
        - [CBOW](#cbow)
        - [GloVe](#glove)
    - [Word Regularities](#regularities)
    - [Doc2vec](#doc2vec)
- [Image Embeddings](#imgemb)
    - [Single layer embeddings](#single-layer-emb)
        - [DeCAF](#decaf)
        - [CNN Features off-the-shell](#feat-shell)
    - [Studies of transferability](#stud-trans)
        - [Transferability of features](#trans-feat)
        - [Factors of transferability](#fact-trans)
    - [Multiple layer embeddings](#mult-layer-emb)
        - [Full-Network embedding](#full-net-emb)
- [Multimodal Embeddings](#mme)
    - [Introduction](#mme:intro)
    - [Image and Text Multimodal Embeddings](#mme:imgtxt)
        - [Two separate embeddings](#mme:2emb)
        - [Pairwise Ranking Loss](mme:rank)
        - [Available datasets for Image Captioning](mme:datasets)
        - [Applications today](#mme:app)
    - [Other multimodal combinations](#mme:other)
- [Bibliography](#bib)


<a name='intro'></a>
## Introduction

An embedding is a transformation of data, used to represent a given data instance through an alternative set of features or characteristics. This set of features defines a representation space, i.e., an embedding space. Simply put, an embedding is a translation from one language (the source data representation, e.g., pixels for an image, or words in text) to a different language (e.g., numerical values). To generate an embedding, all that is needed is a function, which, when applied to a data instance, outputs the instance representation in the embedding space. The resulting embedding space may have a different dimensionality than the source representation space. Notice that an embedding does not require an activation function, as we are not trying to find non-linear features defining the input. Instead, we are just trying to find an alternative representation of a given input which we can tune and adapt for a given task.

The goal of defining and generating embeddings is to obtain representation spaces which satisfy certain desirable properties. For example, that distances within the embedding space are correlated with a similarity of some sort. In the context of Artificial Intelligence, this is interesting as it may result in distributed representations of symbols [1], closing the gap between sensing and thinking. For more details on symbolic vs sub-symbolic AI see [3,4].

In the context of feed-forward neural networks (FNN), embeddings are often explored by using trained neurons as the features defining the embedding space. This may result in continuous, fixed-length vector representations of data instances, which enables multiple applications.


<a name='wordemb'></a>
## Word Embeddings

One of the first approaches to embedding spaces were word embeddings, proposed by Bengio et.al. in [5]. Word embeddings transform words into high-dimensional vectors of numbers, with the goal of solving a simple task. This simple task can be, for example, discriminating between coherent sentences and incoherent ones. The following figure illustrates an example of such architecture. In it, each word is codified as a numerical vector W of fixed length (e.g., 300), which are fed to a classifier module R. The output S would determine, in this particular example, if the sentence is coherent or not. Training a model such as this is easy thanks to the large quantity of data available. One can easily collect millions of coherent sentences from digital sources, and shuffle words of those same sentences to produce incoherent ones.

<div style="text-align:center">
    <img src="/images/wordemb.png" width="400">
</div>
 <div><p style="text-align: center;">Schema of a system for learning word embeddings for a given purpose. Source [6].</p></div>

Although the task is to discriminate coherent sentences through R, what is more interesting of this architecture is the vector representation that are being learnt during training. W defines an embedding space with a fixed number of dimensions (equal to the length of W). The vector learnt for each word corresponds to the representation of that word within the embedding space.

After thorough training, such an embedding space contains a few interesting properties. For example, given a word v and its vector representation w, the words that have representations closer to w using the Euclidean metric are semantically close.

<div style="text-align:center">
    <img src="/images/wordtable.png" width="500">
</div>
 <div><p style="text-align: center;">Nearest neighbours of words according to vector embeddings learnt by a neural net. Source [7].</p></div>

During training, the word embeddings W converge towards a language that describes which words are coherent in which contexts. Given the limited amount of dimensions (e.g., 300) when compared to the size of the input (thousands of words), each of those features is optimized to capture a portion of the complex information that defines textual coherency.

<div style="text-align:center">
    <img src="/images/worddistrep.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of how an embedding space captures word semantics. Source [11].</p></div>


<a name='word2vec'></a>
### word2vec

The most popular of word embedding approaches is the word2vec model. Word2vec was first proposed in [8], and its implementations were further detailed in [9]. The properties of word2vec were explored by the same authors in [10]. Word2vec includes various implementations of word embeddings. In here we will detail the two main ones: the Continuous Skip-gram model and the Continuous Bag-of-Words (CBOW) model. For more details, see [8,9,10,12,13,14].

<a name='skipgram'></a>
#### Skip-gram model

The skip-gram model generates a vector representation of words to capture the context in which each word appears. In this model, the goal of the training is to learn the probability of a context of words given a source word. This context of words is defined using a sliding window of fixed length through a large corpus of text. 
 
<div style="text-align:center">
    <img src="/images/skipgram_data.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the training data for the skip-gram model. Source [12].</p></div>

In the skip-gram model, each source word is inputed as a one-hot vector, and the output is a list (of length equal to the sliding window) of multinomial distributions. These distributions correspond to the expected probability of words appearing in the context. Training such a model is expensive, due to the number of input variables (as many as words) and the number of training examples. In [9] the authors of word2vec provides some tricks to speed up training, such as negative sampling and subsampling frequent words (see Part 2 of [12] for an explanation). See also [15] for a commented version of the original skip-gram code.

<div style="text-align:center">
    <img src="/images/skip-gram.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the skip-gram model. Source [11].</p></div>


In this model, words are inputed as one-hot vectors. Thus, each word will have a unique vector representation in the hidden layer of the skip-gram model. After training, the weights of the hidden layer corresponding to one particular word represent their embedding representation. See [16] for more details on the skip-gram model.


<a name='cbow'></a>
#### Continuous Bag of Words (CBOW)

The CBOW model is similar to the skip-gram model, but it uses an inverted training scheme. The training purpose of CBOW is to learn the probability distribution of words given their context. As such, the input of the model is a list of unordered words which define a context (hence bag of words), and the output is the word missing in the middle of this context. Similarly to how skip-gram training data is generated, CBOW contexts are also obtained using a sliding window.

<div style="text-align:center">
    <img src="/images/cbow.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the CBOW model. Source [11].</p></div>

The input of the CBOW model is a set of one-hot vectors. The output is a single multinomial distribution, trying to capture the probability of words being found within the context defined by the input. Simply put, while the skip-gram model tries to model words based on their context, the CBOW model tries to model context based on words. However, the goal of both models is the same; to learn dense and rich representations of words. See [17] for more details on the CBOW model.

According to the authors of word2vec, skip-gram works well with little training data, and represents well rare words. CBOW is faster to train and has slightly higher accuracies for frequent words.

<a name='glove'></a>
#### GloVe

Following word2vec, Pennington et.al. published GloVe [19], a method to learn word embeddings following the same goal than word2vec, but using matrix factorization methods, which is more computationally efficient. Additionally, GloVe takes into account full co-occurrence information, instead of a sliding window; it builds a full co-occurrence matrix prior to the learning phase. Authors claim, somewhat controversially, that this provides a boost in performance. See [18] for more details on the differences. [20] provides another explanation on the difference, and includes code to train both GloVe and word2vec. See [21] for the original sources of GloVe, pre-trained word vectors and some further details from the authors.

<a name='regularities'></a>
### Word Regularities

Word embeddings generate representation spaces which encode certain semantics. The most basic of those semantics (as illustrated before) can be explored through distance measures; words with similar vectors (e.g., according to Euclidean measure) correspond to words with similar semantics. However, vector arithmetics can also be used to find offsets (i.e., vector differences) which correspond to semantic regularities.

<div style="text-align:center">
    <img src="/images/reg_capitals.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the regularities between countries and capital cities, using skip-gram. Source [9].</p></div>

By combining both vector arithmetics, and distance, we can find the closest words or phrases to the addition of two words. First we add the vector representation of those two words, and then we find which is the closest (word) vector representation to the result. 

<div style="text-align:center">
    <img src="/images/reg_comp.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the vector compositionality. Source [9].</p></div>

Gender regularities also emerge. Using the representation of both Man and Woman, we can swap the gender of any word, just by subtracting the vector representation of Men and adding the vector representation of Women. This results in the famous equation "King - Man + Woman = Queen", which is only the result of the analogy "King - Man = Queen - Woman".

<div style="text-align:center">
    <img src="/images/queen2.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of gender regularities. King is to Queen like Man is to Woman. Source [24].</p></div>

In fact, all sort of regularities are found within word embedding spaces: People's professions, countries' presidents, chemicals' symbols, companies' products, and even countries' popular foods.

<div style="text-align:center">
    <img src="/images/reg_all.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of all sorts of regularities. Source [9].</p></div>

Word embeddings can also be used for automatic translation [23], as presented by the same authors of word2vec.

<div style="text-align:center">
    <img src="/images/translation.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of language translation. Source [23].</p></div>


<a name='doc2vec'></a>
### Doc2vec

The doc2vec model [25] (also known as Paragraph Vector) extends the word embedding approach to learn representations of blocks of text (e.g., sentences, paragraphs or documents). To do so, it correlates labels and words (unlike word2vec which correlates words with words). In the BOW model of word2vec, vector representations of words are concatenated to predict the next word. In doc2vec, blocks of text are also assigned a unique representation, and are also concatenated to predict following words. Sets of unordered words are fed to the system, with the corresponding text block representation fixed. As such, the text block representation will eventually learn to capture part of the topic information of that block of text. The first of the doc2vec models strongly resembles the BOW architecture, and is called the Distributed Memory Model of Paragraph Vectors (PV-DM).

<div style="text-align:center">
    <img src="/images/pv-dm.png" width="500">
</div>
 <div><p style="text-align: center;">Scheme of the Distributed Memory Model of Paragraph Vectors. Source [25].</p></div>

By inputing both the representations of text blocks and words, PV-DM learns learns representations for words and blocks simultaneously. Once all representations are learnt, this model can be used to generate representations for new blocks of text. By fixing the word representations and training the new block vector until convergence through gradient descent. The document embedding space defined by the representations of the blocks of text can be used to detect similarities between documents.

The PV-DM model take into consideration the word order within the documents, as only consecutive words in the document are used (unordered) to train the representation of the document. A second proposed doc2vec model does not consider word order, instead it samples random words from the document and tries to predict those words. This scheme, know as Distributed Bag of Words model of Paragraph vector (PV-DBOW) looks very similar to the original skip-gram model.


<div style="text-align:center">
    <img src="/images/pv-dbow.png" width="500">
</div>
 <div><p style="text-align: center;">Scheme of the Distributed Bag of Words Model of Paragraph Vectors. Source [25].</p></div>

Doc2vec authors recommend to use a concatenation of both models for building paragraph vectors: Both the PV-DM and the PV-DBOW. However their acknowledge that PV-DM is the best model on its own.



<a name='imgemb'></a>
## Image Embeddings

Word embeddings are trained using a one-hot vector encoding for words. This means that each word is characterized as a bit within a vector of fixed length, as long as the size of the dictionary. For images this is an unfeasible approach, as the number of different images is virtually infinite. Currently, image embeddings are generated by extracting the activations on pre-trained CNNs. Simply put, given an image and a pre-trained CNN, a forward pass of the image through the CNN is made. The vector representation of the image is generated from the activations of a set of neurons from within the CNN.

<div style="text-align:center">
    <img src="/images/image-embedding-space.png" width="800">
</div>  
 <div><p style="text-align: center;">Image embedding space encoding a rich visual representation language.</p></div>

The motivation behind image embedding is as follows. Given a complex vision challenge with a large training set (e.g., ImageNet2012 with its 1,000 classes) and a powerful deep learning model (e.g., VGG19 architecture, with its 19 layers of depth), the model resulting from such training should contain a large and rich visual representation language. A language that could be used for other problems beyond the original training purpose (e.g., classifying indoor scenes), just by training and applying a non-deep learning classifier (e.g., a SVM) using the obtained vector representation. This process of reusing knowledge learnt in one model for another task is known as transfer learning. In fact, this is known as an specific kind of transfer learning defined as feature representation transfer by Pans et al. [26] (although in the literature, some people call it transfer learning for feature extraction). Pans et al. defined as source task the one used to originally train the model, and target task the one that takes advantage from knowledge already learnt by the trained model.

<div style="text-align:center">
    <img src="/images/pans_transfer_learning.png" width="500">
</div>  
 <div><p style="text-align: center;">General scheme of transfer learning procedure. Source [26].</p></div>

<a name='single-layer-emb'></a>
### Single layer embeddings

<a name='decaf'></a>
#### DeCAF

One of the first works exploring the extraction and reuse of deep network activations was DeCAF [27]. In this work, the AlexNet architecture (which includes 5 convolutional layers and 3 fully-connected) is trained for the ImageNet 2012 challenge (source task). Authors feed-forward pass images from target datasets through the pre-trained AlexNet architecture, extracting activations from previous to last layer (just before logits) as the new representation of images. These activations are an embedding representation that encodes image information based on the visual language learnt by the model. One of the target datasets used is the SUN-397 dataset, which contains scene images. Qualitatively, we can evaluate the advantage of encoding images based on the visual language previously learnt in the figure below. Classes are grouped quite well only by using the proposed encoding.

<div style="text-align:center">
    <img src="/images/decaf_img_emb_tsne.png" width="500">
</div>  
 <div><p style="text-align: center;">Images from SUN-397 dataset colored based on their class. Their position corresponds to the embedding space previously learnt. Source [27].</p></div>

The new representation of instances and its corresponding labels are used to feed the target dataset into simple linear classifier (SVM and/or Logistic Regression). Authors evaluate the process on 4 datasets, one for each of the following domains: object recognition, domain adaptation, subcategory recognition and scene recognition. Results showed that proposed methodology outperformed previous state-of-the-art approaches based on multi-kernel learning techniques with traditional hand-crafted features.

<div style="text-align:center">
    <img src="/images/decaf_vs_surf.png" width="800">
</div>  
 <div><p style="text-align: center;">Images from scissors class in the embedding space learnt by DeCAF against the embedding space generated with hand-crafted SURF features. Source [27].</p></div>

<a name='feat-shell'></a>
#### CNN Features off-the-shell

A similar work titled "CNN Features off-the-shell" [28] was published the same year with the goal of extending the research line. Authors decided to make a deeper study of the methodology proposed by Donahue et al. [27], mainly by increasing the experimental evidences. The methodology proposed is analogous, using the Overfeat architecture (formed by 6 convolutional layers and 3 fully-connected), training on the ImageNet2012 dataset, using as embedding representation the activations from a fully-connected layer, and training a simpler linear SVM model for the target task. Nevertheless, authors included a couple of new variables to the process: usage of data augmentation (cropping and rotating samples) and the L2 normalization of the embedding representations.

<div style="text-align:center">
    <img src="/images/feature_extraction.png" width="800">
</div>  
 <div><p style="text-align: center;">Scheme of transfer learning for feature extraction following methodology from [28].</p></div>

In this second work, authors report results of the proposed technique over 11 datasets divided in 4 domains: Image classification, Object detection, Attribute detection and Visual instance retrieval. Reporting results of this methodology on 11 datasets instead of the previous 4 datasets supposes an important extension on experiments, providing more insights on the goodness of using this methodology (in cases where datasets have not enough images for training deep models from scratch) against using hand-crafted features approaches. Additionally, authors included a brief study about the usage of different layers to obtain the image embedding, where they conclude that last layers encode more appropriate image representations for applying transfer learning for feature extraction.

<a name='stud-trans'></a>
### Studies of transferability

<a name='trans-feat'></a>
#### Transferability of features

On the same research line of transfer learning, Yosinki et al. [29] published a work where they study the effect of another specific kind of transfer learning called transfer learning for fine-tunning. This kind of transfer learning is based on the idea of partially reuse a pre-trained deep model by keeping weights from some beginning layers and randomizing the rest of layer weights. So, layers that have not been randomized may encode a rich visual language that might be of interest for the new target task, avoiding to learn it again from scratch. In this work, authors study the impact of choosing a different number of last layers to randomize, keeping more visual language or less. This might seem an irrelevant decision by itself, but it is in fact a trade-off considering the fact that the visual language learnt in first layers is generic to the nature of images and the one in last layers is much more specific to the source tasks.

<div style="text-align:center">
    <img src="/images/fine_tunning_layers_study.png" width="600">
</div>  
 <div><p style="text-align: center;">Accuracy results of different fine-tuned models. A and B are tasks (e.g., ImageNet2012). baseX refers to training the deep architecture from scratch using task X. XnY+ refers to the process of training the deep architecture using task Y, then keep the weights from n first layers and randomize the rest and, finally re-train the architecture using task X. If XnY does not have the + sign, it means that n first layer weights are frozen and can not be modified afterwards by the re-training process using task X. For more information check source [29].</p></div>

<a name='fact-trans'></a>
#### Factors of transferability

Going back to transfer learning for feature extraction, one interesting work was published in 2015 [30]. Authors of this work argue that usually we train a deep architecture to maximize the performance on a source task, however, someone could want to maximize the performance of the embedding representations for a particular target task. In that case, how should we train those embedding representations using source task to allow better representations for an specific target task?

In order to bound the problem, authors define a set of factors that affect the transferability of representations between specific source and target tasks. These factors go from the deep architecture used, passing through the decision of applying or not early stopping, to making use of dimensionality reduction on the resulting embedding representation, among some other factors defined in [30].

<div style="text-align:center">
    <img src="/images/factors_transferability.png" width="700">
</div>  
 <div><p style="text-align: center;">Scheme of transfer learning for feature extraction and some detailed factors of transferability defined at [30].</p></div>

In this work, authors make use of 17 datasets to demonstrate, through experimental evidences, how factors should be set given a source and target task (i.e., labeled dataset different from the one used to pre-train the deep model). These 17 datasets are ordered by authors based on the similitude with source task used (i.e., ImageNet2012). Based on how far the target task is from source task, authors also generalize an interesting table defining best practices relating some transfer factors.

<div style="text-align:center">
    <img src="/images/factors_datasets.png" width="1000">
</div>  
 <div><p style="text-align: center;">Range of 15 target tasks sorted categorically by their similarity to ImageNet12 object image classification task. Source [30].</p></div>

<div style="text-align:center">
    <img src="/images/factors_table.png" width="500">
</div>  
 <div><p style="text-align: center;">Best practices to transfer a ConvNet representation trained for the source task of ImageNet to a target tasks summarizing some factors. Source [30].</p></div>

<a name='mult-layer-emb'></a>
### Multiple layer embeddings

<a name='full-net-emb'></a>
### Full-Network embedding

Last but not least, a recent work proposed to build an image embedding representation that extracts information from the entire network. While previous contributions to feature extraction propose embeddings based on a single layer of the network, in this work, authors propose a full-network
embedding which successfully integrates convolutional and fully connected features, coming from all layers of a deep convolutional neural network.

Since extracting all activations directly from convolutional layers is unfeasible due to high dimensionality, authors propose to apply an average pooling to reduce the positional dimensionality of convolutional feature activations to a single value. Then, convolutional layer activations and fully-connected ones are concatenated forming a raw embedding with a  reasonable dimensionality that encodes the visual language already learnt by the pre-trained deep model.

<div style="text-align:center">
    <img src="/images/fne_scheme.png" width="800">
</div>  
 <div><p style="text-align: center;">Overview of the proposed out-of-the-box full-network embedding generation workflow. Source [31].</p></div>

Full-networks embedding introduce the leverage of information encoded in all features of the CNN (i.e., all features from the raw embedding) based on a couple of techniques, the feature standardization and a novel feature discretization methodology. The former provides context-dependent embeddings, which adapt the representations to the target task. The later reduces noise and regularizes the embedding space while keeping the size of the original representation language.

The resultant full-network embedding is shown to outperform single-layer embeddings in several classification tasks. Specially, experiments show that the full-network is more robust that single-layer embeddings when an appropriate source model is not available.

<a name='mme'></a>
## Multimodal Embeddings 
### Introduction <a name='mme:intro'></a>
Think for a moment on the first time you went to the beach. What comes into your mind? Do you remember the scene?
<div style="text-align:center">
    <img src="/images/beach-1525755_1920.jpg" width="800">
</div>  
 <div><p style="text-align: center;">Browsing for happiness.</p></div>

Do you remember the view?
Did the waves sound loud?
Was the water cold?
Did it taste salty?
How did it smell?
Was it windy?
Was it sunny?
What about the sand?
How did it feel?
Could you get rid of it?

When you think about the concept beach what comes into your mind?
Is it just the image of the beach, or also the feel of the sand and the sound of the sea?

**When our brain creates ideas or concepts usually we combine different channels of perception to form our mental image.** 
In humans the senses convey most of the information.
We talk about the traditional five senses:

1. Sight (vision)
2. Hearing (audition)
3. Taste (gustation)
4. Smell (olfaction)
5. Touch (somatosensation)

But also about not so well known ones:

6. Temperature (thermoception)
7. Kinesthetic sense (proprioception)
8. Pain (nociception)
9. Balance (equilibrioception)
10. Vibration (mechanoreception)
11. Internal stimuly (e.g. the different chemoreceptors for detecting salt and carbon dioxide concentrations in the blood, or sense of hunger and sense of thirst)

**If we want to represent really complex ideas, we need to include in the representation  information from different channels.**

But, can it be the same representation?
This is still an open question and an active field of research.

<a name='mme:imgtxt'></a>
### Image and Text Multimodal Embeddings
We can understand the language as a channel of information. Actually, the channel would be the stream of sound, or the visual input of characters, but we can take a shortcut and use the string of characters itself as the input in the same way we use image pixel values as a representation of a visual input without getting into details on human visual information preprocessing.

The input for an image and text multimodal embedding is a pair of an image and an associated string of text. The output would be a representation where both inputs are represented as a unique entity (or close enough). This common representation is (as usual) a multidimensional vector (or a point in the multimodal embedding space).

Given that an image and a text are different modalities of information, their original representations are different as well. An image is codified as a 3D vector of pixel intensity values and the text as a string of characters. Thus, different embeddings are required to map different modalities to a common embedding space.

The most common problem tackled in this setting is the **image captioning**. In this problem the input we have is an image and a short caption describing the image. We want to represent both as a unique (or close enough) vector. To achieve it we train one embedding for the image and another embedding for the caption. We want hose embeddings to output the same (or similar) point in the multimodal embedding space while keeping unrelated images or captions embeddings far away. In order to achieve this representation we use a specific loss function: [Pairwise Ranking Loss](#mme:rank).

<div style="text-align:center">
    <img src="/images/mme_space.png" width="800">
</div>  
 <div><p style="text-align: center;">Multimodal embedding.</p></div>


<a name='mme:2emb'></a>
#### Two separate embeddings
As introduced before, the most common way to learn a multimodal embedding is to learn two separate embeddings, for image and text. Using embeddings based on Neural Networks is very convenient since the error computed in the Ranking Loss imposed to the multimodal embedding can be easily back-propagated to the NN parameters of both embeddings.

<div style="text-align:center">
    <img src="/images/img-txt1.png" width="800">
</div>  
 <div><p style="text-align: center;">Example of image and text embeddings to form a multimodal embedding[41].</p></div>

These embeddings can be fully trained on the image captioning dataset or only partially. Given the sizes of the [publicly available datasets](#mme:datasets) on Image Captioning (and the difficulty on building one) usually the image embedding is pre-trained on an image classification task (ImageNet) and partially or completely fine-tuned on the multimodal embedding. Similarly happens for the text embedding but in this case it is more usual to fully train the embedding since its complexity use to be several orders of magnitude lower.

For **image embeddings** the CNN embedding is the "standard". Different works propose different variations on the previously explained versions of [image embeddings](#imgemb), but all share the same basic methodology:
1. Define a CNN architecture
2. Pre-train the CNN on ImageNet
3. Use CNN features as an initial image embedding. (There will be no more training on CNN)
4. Add a trainable transformation from this image embedding to the multimodal embedding. (Can be one or several NN layer)

<div style="text-align:center">
    <img src="/images/img-txt2.png" width="800">
</div>  
 <div><p style="text-align: center;">Example of image embedding using the Full Network Embedding (FNE) to form a multimodal embedding [41].</p></div>

Most usually in step 3 only the features from the last layer of the CNN are selected, although any kind of representation can be used in this step. In the examples we use the Full Network Embedding (FNE) to form a multimodal embedding [41], a solution shown to improve results over a one-layer CNN embedding.

In step 4 different variants have been proposed. Word2VisualVec [42] uses no transformation in this point so the multimodal embedding space coincide with the visual embedding. The most common approach is to allow some adaptation using an affine transformation (i.e., a fully connected layer without non-linear activation function). Using a transformation allow for different dimensionalities for the multimodal space and the image embedding.

The **text embedding** is conceptually similar to the [Doc2vec](#doc2vec) explained previously but present some additional difficulties:
- The whole text need to be encoded in a single vector
- The length of the text can be variable.
- There may be rare words in the vocabulary used in the text (difficult to learn their meaning from few examples)
- We do not have a large corpus to train on.

Thus, the solutions usually implemented are different. In this case the text embedding is most usually obtained through a Recurrent Neural Network. The text is feeded to the network, one work at every time step and the hidden state of the network at the last time step is used as the text embedding. This approach can deal without problems with texts of different length. Gated Recurrent Units (GRU) Neural Network are a common choice since they obtain a high performance while being less complex than Long Short Term Memories (LSTM).

<div style="text-align:center">
    <img src="/images/img-txt3.png" width="800">
</div>  
 <div><p style="text-align: center;">Example of **image embedding** using a Gated Recurrent Units Neural Network to form a multimodal embedding [41].</p></div>

A previous step required is to map the vocabulary into GRU's input vectors. A lookup table (i.e., a word dictionary with one-hot vector encoding) can be enough, but other encodings, like the ones explained in [word2vec](#word2vec), help to achieve better results.

It is possible to add a transformation between the text embedding of the last hidden state of the RNN and the multimodal embedding. This transformation can be one or several fully connected layers and allow more flexibility in the text embedding part. However, GRU NN already have a high capacity to adapt (and we can increase it increasing the number of neurons) so such a transformation would be mainly motivated to allow for a dimensionality change between GRU hidden state and the MME.

<div style="text-align:center">
    <img src="/images/FN_multimodal.png" width="800">
</div>  
 <div><p style="text-align: center;">Full schema of a multimodal embedding [41]. The parts in orange are trained for the multimodal embedding.</p></div>

<a name='mme:rank'></a>
#### Pairwise Ranking Loss
As introduced before, our objective is to obtain a common representation for inputs from different channels in the multimodal embedding space. To achieve it we define a loss in the MME space that forces associated items to be represented close to each other and not related items to be **farther**. It is important to notice that we do not want not related items to very far away, just farther than the related one. We still want other similarities arising from the embeddings to be encoded in the position of samples in the embedding space. To force it we use a pairwise ranking loss. Following we explain it for the example case of image captioning, for which we have only 2 different channels.

In this case the training procedure consist on the optimization of the pairwise ranking loss between the correct image-caption pair and a random pair. Assuming that a correct pair of elements should be closer in the multimodal space than a random pair. The loss can be formally defined as follows:

<div style="text-align:center">
    <img src="/images/Pairwise_ranking_loss.jpg" width="600">
</div>  
 

Where **_i_** is an image vector, **_c_** is its correct caption vector, and _**i**<sub>k</sub>_ and _**c**<sub>k</sub>_ are sets of random images and captions respectively. The operator _s(·,·)_ defines the cosine similarity. This formulation includes a margin term _alpha_ to avoid pulling the image and caption closer once their distance is smaller than the margin. This makes the optimization focus on distant pairs instead of improving the ones that are already close.

Other losses can be used to define multimodal embeddings with different properties. An interesting work is _Order-Embeddings of Images and Language_ [43]. Here authors propose an **Order Embedding** to represent hierarchy using an order-violation penalty. Authors apply it at the image captioning problem considering that images are in a lower hierarchy level of the corresponding captions.


<a name='mme:datasets'></a>
#### Available datasets for Image Captioning
The most commonly used datasets in image captioning are:

- The [Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html) dataset [37] contains 8,000 hand-selected images from Flickr, depicting actions and events. Five correct captions are provided for each image.
- The [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) dataset [38] is an extension of Flickr8K. It contains 31,783 photographs of everyday activities, events and scenes. Five correct captions are provided for each image.
- The [MSCOCO](http://cocodataset.org/#download) dataset [39] includes images of everyday scenes containing common objects in their natural context. For captioning, 82,783 images and 413,915 captions are available for training, while 40,504 images and 202,520 captions are available for validation. Captions from the test set are not publicly available.
- The [SPEECH-COCO](http://cocodataset.org/#download) dataset [40] is an extension of MSCOCO where speech is added to image and text. Speech captions are generated using text-to-speech (TTS) synthesis resulting in 616,767 spoken captions (more than 600h) paired with images.

Their main drawback is the size of the dataset (compared to the million-images labelled datasets for image classification) and the difficulty to obtain sentences of the same style from crowd sourcing.

<a name='mme:app'></a>
#### Applications today
The multimodal embeddings obtained encode simultaneously the visual information obtained from the image and semantic information obtained from the text. Numerous problems can benefit from this embedding richer than a single image or text embedding. The first and "obvious" application is the symmetric problem of caption/image retrieval. 

<div style="text-align:center">
    <img src="/images/prob2.png" width="400">
</div>  
 <div><p style="text-align: center;">**Caption Retrieval**. A.K.A. Image Annotation.
 For a given image, find the caption that best describe the image from a set of defined captions.</p></div>

<div style="text-align:center">
    <img src="/images/prob1.png" width="400">
</div>  
 <div><p style="text-align: center;">**Image Retrieval**. A.K.A. Image Search.
 For a given caption, find the image that is best described by the caption from a set of given images.</p></div>

Once build the multimodal embedding space representing the query image(caption) and all the captions(images) in the set to look into, we just need to find the nearest neighbour.

It has been shown that multimodal embeddings are capable to encode multimodal linguistic regularities on visual and textual features similar to the [word regularities](#regularities) explained previously.

<div style="text-align:center">
    <img src="/images/car-red.png" width="600">
</div>  
 <div><p style="text-align: center;">Example of multimodal linguistic regularities.</p></div>

Nowadays multimodal embeddings are the first step for several successful **image caption generation** approaches [32],[33],[34]. In this case the multimodal embedding is used as a representation of the query image to feed in a text generation network. Results indicate that linguistic information encoded in this image representation help produce "better" captions.

<div style="text-align:center">
    <img src="/images/Caption_generation_samples.jpg" width="800">
</div>  
 <div><p style="text-align: center;">Example of captions generated from a multimodal embedding [32].</p></div>

<a name='mme:other'></a>
### Other multimodal combinations

Focusing in an extended version of multimodal embeddings, Kaiser et al. [44] present a multimodal setting called MultiModel, that yields good results on a variety of tasks of different nature. Although previous multimodal settings work with 2 modalities, MultiModel works with 4 modalities. It is capable of treating single tasks regarding text, images, audio, categorical data or combinations of them.

Authors follow a 2 steps strategy:

 - First step consist of unifying all nature representations through a set of modality-specific sub-networks. These sub-networks are called modality nets because they are specific to each modality (e.g., image, text or audio) and define transformations between these external domains and a unified representation. Modality nets are designed to be computationally minimal, authors only want to obtain a rich feature representation from these nets, letting the major computation effort on the second step. 

 - Second step consist on a trainable model capable of dealing with all embedding representations as input. This model is formed by three main blocks: Encoder, Input-Output Mixer and Decoder. The Encoder codifies the input embeddings from all modality nets and outputs the encoded input. The Input-Output Mixer takes as input the encoded input and decoded output to evaluate the encoded output. Lastly, the Decoder takes as input the encoded input and encoded output to generate the decoded output. These three blocks architecture is analogous to an state machine, where there is an input managed by the Encoder block, and output managed by the Decoder block and a state managed by the I/O Mixer.

<div style="text-align:center">
    <img src="/images/omlta_overview.png" width="500">
</div>  
 <div><p style="text-align: center;">The MultiModel, with modality-nets, an Encoder, an I/O Mixer, and an autoregressive Decoder. Source [44].</p></div>

The MultiModel core blocks (i.e., Encoder, I/O Mixer and Decoder) are composed by smaller blocks: Convolutional blocks, Attentional blocks and Mixture-of-Experts blocks. All blocks provide a different set of properties. Convolutional and Attentional blocks are trivial: a Convolutional block performs local computation while an Attentional block allows to focus on content based on its position. However, a Mixture-of-Experts block may not be that trivial. It is formed by a set of simple feed-forward neural networks (experts) and a trainable gating network responsible of selecting a sparse combination of experts based on the input. So, this block is responsible of specific knowledge computation.

<div style="text-align:center">
    <img src="/images/omlta_blocks.png" width="1000">
</div>  
 <div><p style="text-align: center;">Architecture of the MultiModel divided by blocks. Source [44].</p></div>

Authors demonstrate, for the first time, that a single deep learning model can jointly learn a number of large-scale tasks from multiple domains. A really interesting feature of that model is that we can extract a 4-modal embedding representation from the encoding block to work with.




<a name='bib'></a>
## Bibliography

[1] [Hinton, Geoffrey E. "Learning distributed representations of concepts." Proceedings of the eighth annual conference of the cognitive science society. Vol. 1. 1986.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.7684&rep=rep1&type=pdf)

[2] [colah's blog: Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)

[3] [http://web.media.mit.edu/~minsky/papers/SymbolicVs.Connectionist.html](http://web.media.mit.edu/~minsky/papers/SymbolicVs.Connectionist.html)

[4] [http://futureai.media.mit.edu/wp-content/uploads/sites/40/2016/02/Symbolic-vs.-Subsymbolic.pptx_.pdf](http://futureai.media.mit.edu/wp-content/uploads/sites/40/2016/02/Symbolic-vs.-Subsymbolic.pptx_.pdf)

[5] [Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of machine learning research 3.Feb (2003): 1137-1155](http://machinelearning.wustl.edu/mlpapers/paper_files/BengioDVJ03.pdf)

[6] [Bottou, Léon. "From machine learning to machine reasoning." Machine learning 94.2 (2014): 133-149.](https://arxiv.org/pdf/1102.1808.pdf)

[7] [Collobert, Ronan, et al. "Natural language processing (almost) from scratch." Journal of Machine Learning Research 12.Aug (2011): 2493-2537.](https://arxiv.org/pdf/1103.0398v1.pdf)

[8] [Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013).](https://arxiv.org/pdf/1301.3781.pdf)

[9] [Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[10] [Mikolov, Tomas, Wen-tau Yih, and Geoffrey Zweig. "Linguistic regularities in continuous space word representations." hlt-Naacl. Vol. 13. 2013.](http://www.aclweb.org/anthology/N13-1090)

[11] [the morning paper - The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)

[12] [Chris McCormick - Word2Vec Tutorial - The Skip-Gram Model (Part 1 & 2)](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

[13] [Rong, Xin. "word2vec parameter learning explained." arXiv preprint arXiv:1411.2738 (2014).](https://arxiv.org/pdf/1411.2738)

[14] [Goldberg, Yoav, and Omer Levy. "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method." arXiv preprint arXiv:1402.3722 (2014).](https://arxiv.org/pdf/1402.3722)

[15] [https://github.com/chrisjmccormick/word2vec_commented](https://github.com/chrisjmccormick/word2vec_commented)

[16] [Alex Minnaar, Word2Vec Tutorial Part I: The Skip-Gram Model](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_I_The_Skip-Gram_Model.pdf)

[17] [Alex Minnaar, Word2Vec Tutorial Part II: The Continuous Bag-of-Words Model](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf)

[18] [the morning paper - GloVe: Global Vectors for Word Representation](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/)

[19] [Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.](http://www.aclweb.org/anthology/D14-1162)

[20] [Radim Rehurek - Making sense of word2vec](https://rare-technologies.com/making-sense-of-word2vec/)

[21] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

[22] [DL4J - Neural Word Embeddings](https://deeplearning4j.org/doc2vec)

[23] [Mikolov, Tomas, Quoc V. Le, and Ilya Sutskever. "Exploiting similarities among languages for machine translation." arXiv preprint arXiv:1309.4168 (2013).](https://arxiv.org/pdf/1309.4168v1.pdf)

[24] [A gentle introduction to Doc2Vec](https://medium.com/towards-data-science/a-gentle-introduction-to-doc2vec-db3e8c0cce5e). An explanation of doc2vec plus an example of addapting the model for a usecase of industry.

[25] [Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents." Proceedings of the 31st International Conference on Machine Learning (ICML-14). 2014.](https://arxiv.org/pdf/1405.4053.pdf)

[26] [Pan, Sinno Jialin, and Qiang Yang. "A survey on transfer learning." IEEE Transactions on knowledge and data engineering 22.10 (2010): 1345-1359.](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)

[27] [Donahue, Jeff, et al. "Decaf: A deep convolutional activation feature for generic visual recognition." International conference on machine learning. 2014.](http://proceedings.mlr.press/v32/donahue14.pdf)

[28] [Sharif Razavian, Ali, et al. "CNN features off-the-shelf: an astounding baseline for recognition." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2014.](http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf)

[29] [Yosinski, Jason, et al. "How transferable are features in deep neural networks?." Advances in neural information processing systems. 2014.](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)

[30] [Azizpour, Hossein, et al. "Factors of transferability for a generic convnet representation." IEEE transactions on pattern analysis and machine intelligence 38.9 (2016): 1790-1802.](https://arxiv.org/pdf/1406.5774.pdf)

[31] [Garcia-Gasulla, Dario, et al. "An Out-of-the-box Full-network Embedding for Convolutional Neural Networks." arXiv preprint arXiv:1705.07706 (2017).](https://arxiv.org/pdf/1705.07706)

[32] [Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel. Unifying visual-semantic embeddings with multimodal neural language models. arXiv preprint arXiv:1411.2539, 2014.](https://arxiv.org/abs/1411.2539)

[33] [Ryan Kiros, Ruslan Salakhutdinov, and Rich Zemel. Multimodal neural language models. In Proceedings of the 31st International Conference on Machine Learning (ICML-14), pages 595–603, 2014.](http://www.cs.toronto.edu/~rkiros/papers/mnlm2014.pdf)

[34] [Qing Sun, Stefan Lee, and Dhruv Batra. Bidirectional beam search: Forward-backward inference in neural sequence models for fill-in-the-blank image captioning. arXiv preprint arXiv:1705.08759, 2017.](https://arxiv.org/abs/1705.08759)

[35] [Benjamin Klein, Guy Lev, Gil Sadeh, and Lior Wolf. Associating neural word embeddings with deep image representations using fisher vectors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4437–4446, 2015.](http://ieeexplore.ieee.org/document/7299073/)

[36] [Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)

[37] [Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. Collecting image annotations using amazon’s mechanical turk. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon’s Mechanical Turk, pages 139–147. Association for Computational Linguistics, 2010.](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html)

[38] [Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014.](http://shannon.cs.illinois.edu/DenotationGraph/)

[39] [Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C Lawrence Zitnick, and Piotr Dollar. Microsoft coco: Common objects in context. arXiv preprint arXiv:1405.0312, 2014.](http://cocodataset.org/#download)

[40] [Havard, William, Laurent Besacier, and Olivier Rosec. "SPEECH-COCO: 600k Visually Grounded Spoken Captions Aligned to MSCOCO Data Set." arXiv preprint arXiv:1707.08435 (2017).](https://arxiv.org/abs/1707.08435)

[41] [Vilalta, Armand, et al. "Full-Network Embedding in a Multimodal Embedding Pipeline." Proceedings of the 2nd Workshop on Semantic Deep Learning (SemDeep-2). 2017.](http://www.aclweb.org/anthology/W/W17/W17-7304.pdf)

[42] [Dong, Jianfeng, Xirong Li, and Cees GM Snoek. "Word2VisualVec: Cross-media retrieval by visual feature prediction." arXiv preprint arXiv:1604.06838 (2016).](https://pdfs.semanticscholar.org/de22/8875bc33e9db85123469ef80fc0071a92386.pdf)

[43] [Vendrov, Ivan, et al. "Order-embeddings of images and language." arXiv preprint arXiv:1511.06361 (2015).](https://arxiv.org/abs/1511.06361)

[44] [Kaiser, Lukasz, et al. "One Model To Learn Them All." arXiv preprint arXiv:1706.05137 (2017).](https://arxiv.org/pdf/1706.05137.pdf)

### Other uncited sources:

[Socher - Stanford CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)

[Doc2vec tutorial using gensim](https://rare-technologies.com/doc2vec-tutorial/)

[Spanish Billion Word Corpus and Embeddings](http://crscardellino.me/SBWCE/)

