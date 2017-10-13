---
permalink: /emb-space-theory/
---

This page contains the theoretical part of the Embedding Spaces topic for the Deep Learning course at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya. 

Table of Contents:

- [Introduction](#intro)
- [Word Embeddings](#wordemb)
    - [word2vec](#word2vec)
- [Bibliography](#bib)


<a name='intro'></a>
## Introduction

An embedding is a transformation of data, used to represent a given data instance through an alternative set of features or characteristics. This set of features defines a representation space, i.e., an embedding space. Simply put, an embedding is a translation from one language (the source data representation, e.g., pixels for an image) to a different language (the target data representation). To generate an embedding, all that is needed is a function, which, when applied to a data instance, outputs the instance representation in the embedding space. The resulting embedding space may have a different dimentionality than the source representation space.

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

After thorough training, such an embedding space containst a few interesting properties. For example, given a word v and its vector representation w, the words that have representations closer to w using the Euclidean metric are semantically close.

<div style="text-align:center">
    <img src="/images/wordtable.png" width="500">
</div>
 <div><p style="text-align: center;">Nearest neighbors of words according to vector embeddings learnt by a neural net. Source [7].</p></div>

During training, the word embeddings W coverge towards a language that describes which words are coherent in which contexts. Given the limited amount of dimensions (e.g., 300) when compared to the size of the input (thousands of words), each of those features is optimized to capture a portion of the complex information that defines textual coherency.

<div style="text-align:center">
    <img src="/images/worddistrep.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of how an embedding space captures word semantics. Source [11].</p></div>


<a name='word2vec'></a>
### word2vec

The most popular of word embedding approaches is the word2vec model. Word2vec was first proposed in [8], and its implementations were further detailed in [9]. The properties of word2vec were explored by the same authors in [10]. Word2vec includes various implementations of word embeddings. In here we will detail the two main ones: the Continuous Skip-gram model and the Continuous Bag-of-Words (CBOW) model. For more details, see [8,9,10,12,13,14].

#### Skip-gram model

The skip-gram models generates a vector representation of words to capture the context in which each word appears. In this model, the goal of the training is to learn the probability of a context of words given a source word. This context of words is defined using a sliding window of fixed length through a large corpus of text.
 
<div style="text-align:center">
    <img src="/images/skipgram_data.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the training data for the skip-gram model. Source [12].</p></div>

In the skip-gram model, each source word is inputted as a one-hot vector, and the output is the probability

<div style="text-align:center">
    <img src="/images/skip-gram.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the skip-gram model. Source [11].</p></div>


<div style="text-align:center">
    <img src="/images/cbow.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the CBOW model. Source [11].</p></div>


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

[13] [Rong, Xin. "word2vec parameter learning explained." arXiv preprint arXiv:1411.2738 (2014).] (https://arxiv.org/pdf/1411.2738)

[14] [Goldberg, Yoav, and Omer Levy. "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method." arXiv preprint arXiv:1402.3722 (2014).] (https://arxiv.org/pdf/1402.3722)



### Other uncited sources:

