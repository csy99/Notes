# fastText

由Facebook主持的研究。该研究分为两个项目：子词嵌入和高效文本分类。有学者认为fastText只包含高效文本分类，主张它才是该研究的重中之重。不过，在Facebook相关研究的[官网](https://research.fb.com/downloads/fasttext/)，是对两个项目各给出一篇论文的链接。本文以官网为准，对两个模块都进行介绍。

## 子词嵌入

构词学（morphology）作为语言学的一个重要分支，研究的正是词的内部结构和形成方式。英语单词通常有其内部结构和形成方式。例如，我们可以从“dog”, “dogs” 和 “dogcatcher” 的字面上推测它们的关系。这些词都有同一个词根“dog”，但使用不同的后缀来改变词的含义或者词性。这一特点不是英语所特有的，欧洲语系中很多语言都具有这一特点。

鉴于此，fastText提出了子词嵌入(subword embedding)的方法，从而试图将构词信息引入word2vec中的跳字模型。

在fastText中，每个中心词被表示成子词的集合。下面我们用单词“where”作为例子来了解子词是如何产生的。首先，我们在单词的首尾分别添加特殊字符“\<”和“\>”以区分作为前后缀的子词。然后，将单词当成一个由字符构成的序列来提取n元语法。例如，当n=3时，我们得到所有长度为3的子词：“\<wh”， “whe”， “her”， “ere”， “re\>”以及特殊子词“\<where\>”。

在fastText中，对于一个词$w$，我们将它所有长度在3∼6的子词和特殊子词的并集记为$G_w$。那么词典则是所有词的子词集合的并集。假设词典中子词$g$的向量为$z_g$，那么跳字模型中词$w$的作为中心词的向量$v_w$则表示成
$$
v_w = \sum_{g\in G_w} z_g
$$
我们可以的到以下这个得分函数
$$
s(W, c) = \sum_{g\in G_w} z_g^Tv_c
$$
与跳字模型相比，fastText中词典规模更大，造成模型参数更多，同时一个词的向量需要对所有子词向量求和，继而导致计算复杂度更高。为了缓解空间复杂度的压力，该模型使用可以将n元语法映射到整数1~K的哈希函数。该函数是Fowler-Noll-Vo hashing function (更精确地说是the FNV-1a变种)。哈希到同一个位置的多个子词的n元语法是会共享一个词向量的。具体可以参阅[Feature hashing for large scale multitask learning](https://alex.smola.org/papers/2009/Weinbergeretal09.pdf)。

但与此同时，较生僻的复杂单词，甚至是词典中没有的单词，可能会从同它结构类似的其他词那里获取更好的词向量表示。

## 高效文本分类

看到这里，估计大家有疑问了。子词嵌入增加了计算复杂度，怎么还叫"fastText"呢？这里面的"fast"其实说的就是高效文本分类。这一款文本分类与向量化工具与其他已有模型对比，最大的特点就是快（数据截止论文发出之时，也就是2016年）。

<img src="https://i.postimg.cc/Yq3zRCRr/fast-Text-training-time.png" height=200>

什么TMD叫惊喜！如果线性模型具有秩约束和快速的损失近似值，那么在保证fastText达到其他模型的最佳性能的前提下，能在10分钟内训练十亿词级别语料库的词向量。

细往下看，其实fastText使用的很多技巧在当时早已为人所熟知。

比如层序softmax。这个技巧我们已经在word2vec中讲过了，在此不赘述。假设我们有$k$类文档，每个文档的词向量维度是$h$。这个技巧使得文档分类的复杂度从$O(kh)$降到了$O(h log(k))$。

另外，fastText和word2vec中的词袋模型(CBOW)非常相似。结构如下图。

<img src="https://i.postimg.cc/W3KjZVFT/fast-Text-arch.png" height=260>

单词的向量表示会被综合考虑形成一个文本向量表示（将整篇文档的词及子词嵌入向量叠加平均得到文档向量），输入到一个线性模型当中。文档总数为$N$，激活函数$f$为softmax，权重矩阵$A$是单词的查找表，$x_i$表示第$i$个文档被标准化之后的特征表示。模型有负对数损失如下。
$$
-\frac 1 N \sum_{i=1}^N y_i log(f(BAx_i))
$$
需要注意的是，在word2vec中词向量的生成是无监督，而fastText主要用于文本分类，所以词向量的训练过程是监督学习。





# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第10章
- [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf), Piotr Bojanowski and Edouard Grave and Armand Joulin and Tomas Mikolov
- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf), Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov