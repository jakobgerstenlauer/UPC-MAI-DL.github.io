---
permalink: /emb-space-theory/
---

This page contains the theoretical part of the Embedding Spaces topic for the Deep Learning course at the Master in Artificial Inteligence of the Universitat Politècnica de Catalunya. 

Table of Contents:

- [Introduction](#intro)
- [Word Embeddings](#wordemb)
    - [word2vec](#word2vec)
        - [skipgram](#skipgram)
        - [CBOW](#cbow)
        - [GloVe](#glove)
    - [Word Regularities](#regularities)
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

<a name='skipgram'></a>
#### Skip-gram model

The skip-gram model generates a vector representation of words to capture the context in which each word appears. In this model, the goal of the training is to learn the probability of a context of words given a source word. This context of words is defined using a sliding window of fixed length through a large corpus of text. 
 
<div style="text-align:center">
    <img src="/images/skipgram_data.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the training data for the skip-gram model. Source [12].</p></div>

In the skip-gram model, each source word is inputted as a one-hot vector, and the output is a list (of length equal to the sliding window) of multinomial distributions. These distributions correspond to the expected probability of words appearing in the context. Training such a model is expensive, due to the number of input variables (as many as words) and the number of training examples. In [9] the authors of word2vec provides some tricks to speed up training, such as negative sampling and subsampling frequent words (see Part 2 of [12] for an explanation). See also [15] for a commented version of the original skip-gram code.

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

Following word2vec, Pennington et.al. published GloVe [19], a method to learn word embeddings following the same goal than word2vec, but using matrix factorization methods, which is more computationally efficient. Additionally, GloVe takes into account full co-occurrence information, instead of a sliding window; it builds a full co-occurrence matrix prior to the learning phase. Authors claim, somewhat controversially, that this provides a boost in performance. See [18] for more details on the differences. [20] provides another explanation on the difference, and includes code to train both GloVe and word2vec. See [21] for the original sources of GloVe, pretrained word vectors and some further details from the authors.

<a name='regularities'></a>
### Word Regularities

Word embeddings generate representation spaces which encode certain semantics. The most basic of those semantics (as illustrated before) can be explored through distance measures; words with similar vectors (e.g., according to Euclidean measure) correpond to words with similar semantics. However, vector arithmetics can also be used to find offsets (i.e., vector differences) which correspond to semantic regularities.

<div style="text-align:center">
    <img src="/images/reg_capitals.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the regularities between countries and capital cities, using skip-gram. Source [9].</p></div>

By combining both vector arithmetics, and distance, we can find the closest words or phrases to the addition of two words.

<div style="text-align:center">
    <img src="/images/reg_comp.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of the vector compositionality. Source [9].</p></div>

Gender regularities also emerge. This results in the famous equation "King - Men + Women = Queen"

<div style="text-align:center">
    <img src="/images/queen.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of gender regularities. Source [10].</p></div>

In fact, all sort of regularities are found within word embedding spaces: People's professions, countries' presidents, chemicals' symbols, companies' products, and even countries' popular foods.

<div style="text-align:center">
    <img src="/images/reg_all.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of all sorts of regularities. Source [9].</p></div>

Word embeddings can also be used for automatic translation, as presented by the same authors of word2vec.

<div style="text-align:center">
    <img src="/images/translation.png" width="500">
</div>
 <div><p style="text-align: center;">Illustration of language translation. Source [23].</p></div>


## Doc2vec

The doc2vec model extends the word embedding approach to learn representations of large blocks of text (e.g., paragraphs or documents). To do so, it correlates labels and words (unlike words and words, like word2vec).


Doc2Vec learns representations for words and labels simultaneously. 



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

### Other uncited sources:

[Socher - Stanford CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)

[Doc2vec tutorial using gensim](https://rare-technologies.com/doc2vec-tutorial/)

[Spanish Billion Word Corpus and Embeddings](http://crscardellino.me/SBWCE/)
