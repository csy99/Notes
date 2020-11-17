更新：
9/4/20
做了关于SG模型最后计算loss的一些补充。对doc2vec损失计算部分出现的错误进行了订正。

11/16/20

补充了部分近似训练的内容

# Represent the Meaning of a Word

### WordNet

WordNet is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations. 

**pros**

1. can find synonyms. 方便寻找同义词

**cons**

1. missing new words (impossible to keep up to date). 缺少新词。
3. subjective. 主观化。
4. requires human labor to create and adapt. 需要耗费大量人力去整理。

### One Hot Encoding

Discrete representation. 

**cons**

1. dimension is extremely high. 维度爆炸。
2. hard to compute accurate word similarity (all vectors are orthogonal). 无法计算词语相似度。

### Bag of Words

Co-occurrence of words with variable window size. 

**cons**

1. dimension is extremely high, will grow as dictionary grows. 维度爆炸，而且会随着字典大小的增大而增大，对下游的ML模型产生影响。

### Word2vec

A neural probabilistic language model. 

Distributional similarity based representations. Represent a word by means of its neighbors.上下文足以辅助理解一个词的意思。

We will build a dense vector for each word type, chosen so that it is good at predicting other words appearing in its context. 

*Distributional similarity & Distributed representation (dense vector)*

There are certain differences between the two. The Distributional Similarity emphasizes that the meaning of a word shall be inferred from its context. Distributed Representation is opposite to One Hot Encoding, and vector representation is non-sparse. 两者有一定的区别。distributional similarity强调能够用上下文去表示某一个单词的意思，而distributed representation与one hot encoding相对，强调向量的表示是非稀疏的。

**pros**

1. can compute accurate word similarity. 可以计算词语相似度。

**cons**

1. The calculation is related word vector instead of semantic word vector, so the polysemous case cannot be solved (1 vector for each word instead of each meaning). 计算出来的是关联词向量，而不是语义词向量，所以无法解决一词多意的情况(每个单词而不是每个词意对应1个向量)。



# 损失函数Loss Function

Softmax function: map from $R^v$ to a probability distribution(从实数空间到概率分布的标准映射方法)。公式分子部分保证将这个数转化成一个正数，分母部分保证所有概率之和为1。

$p_i = \frac {exp(u_i)} {\sum_{j} exp(u_j)}$

我们在求出center/context word的概率分布之后，还需要使用交叉熵来得到loss。

$L(\hat y, y) = −  \sum_{j=1}^V y_j log(\hat y_j)$. 根据公式，在完美预测的情况下，loss是0。



此处以skip-gram的训练为例。

$J = 1 - p(w_{-t} | w_t)$

$w_{-t}$代表$w_t$的上下文（负号表示除了该词之外）。

$p(o|c) = \frac {exp(u_o^T v_c)} {\sum_{w=1}^V exp(u_w^T v_c)}$

o is the outside (or output) word index, c is the center word index. $v_c$ and $u_o$ are center and outside vectors of indices c and o. Softmax uses word c to obtain probability of word o. 

According to this formula, the words in the text will be represented by two vectors. There's one when it's a center word, and there's another when it's a context. 根据这个公式，文中的单词会有两个向量表示。当它作为中心词的时候有一个，当它作为上下文的时候又有一个。

loss的推导过程主要运用softmax的概率分布公式和微积分的链式法则。
$$
\frac{\part}{\part v_c} p(o|c) \\\\
= \frac{\part}{\part v_c} log[exp(u_o^T v_c) / \sum_{w=1}^V exp(u_w^T v_c)] \\\\
= \frac{\part}{\part v_c} log[exp(u_o^T v_c)] ① - \frac{\part}{\part v_c} log[\sum_{w=1}^V exp(u_w^T v_c)] ② \\\\
= u_o - \sum_{x=1}^V p(x|c)u_x
$$
① 表示的是observation，也就是context word实际是什么(true label)。② 表示的是expectation，也就是模型认为概率最高的应该是哪个词(prediction label)。所以，实际上我们就是希望最小化实际和预测之间的差值。
$$
② = \frac{\part}{\part v_c} log[\sum_{w=1}^V exp(u_w^T v_c)] \\\\
= \frac{1}{\sum_{w=1}^V exp(u_w^T v_c)} \frac{\part}{\part v_c} [\sum_{x=1}^V exp(u_x^T v_c)]  \\\\
= \frac{1}{\sum_{w=1}^V exp(u_w^T v_c)} \sum_{x=1}^V [\frac{\part}{\part v_c} exp(u_x^T v_c)]  \\\\
= \frac{1}{\sum_{w=1}^V exp(u_w^T v_c)} \sum_{x=1}^V [exp(u_x^T v_c) \frac{\part}{\part v_c}(u_x^T v_c)]  \\\\
= \frac{1}{\sum_{w=1}^V exp(u_w^T v_c)} \sum_{x=1}^V [exp(u_x^T v_c) u_x]  \\\\
= \sum_{x=1}^V \frac{exp(u_x^T v_c)}{\sum_{w=1}^V exp(u_w^T v_c)} u_x  \\\\
= \sum_{x=1}^V p(x|c) u_x
$$

当我们使用sgd进行优化的时候，每一个窗口最多有2m+1个单词，所以$\nabla_{\theta} J_{t}(\theta)$ 会非常稀疏。



# 训练方法

训练方法(Training Algorithms)包括两种：跳字模型和连续词袋模型。

## 跳字模型

英文是Skip-grams (SG)。Predict context words given target (position independent). 

<img src="https://i.postimg.cc/Y24FPnyX/w2v-context.png" height="200">

在这个案例中，"into"是target (center word)，而"problems turning"和"banking crises"是我们的output context words。假设我们的句子一共有T个单词。我们定义window size（也就是预测上下文的半径）为m，这个案例中m=2。

<img src="https://i.postimg.cc/j2JRhJcX/w2v-pair.png" height = "300">

通过center word和context word组成一组训练数据，喂给word2vec模型。

### 目标函数Objective Function

给定当前中心词时，最大化上下文词的概率。 $\theta$ 代表我们需要优化的参数。给定一个长度为$T$的文本序列。窗口大小是m。 

$$
J'(\theta) = \prod_{t=1}^T \prod_{j=-m,j \ne 0}^m p(w_{t+j}|w_t; \theta)
$$
我们使用负对数似然将目标函数转化为损失函数。

$$
J(\theta) = -\frac 1 {T} \sum_{t=1}^T \sum_{j=-m,j \ne 0}^m log p(w_{t+j}|w_t)
$$

### 训练过程Training Process

<img src="https://i.postimg.cc/FHxkS27K/w2v-skipgram.jpg" height="500">

这张图第一眼看上去非常花哨，但是其实把这个工作流程说清楚了。d表示向量的维度，V是vocabulary size。

图中的$W$是center word矩阵，以列为单位存储每一个单词作为center word的向量表示，$W \in R^{d*V}$。在一个训练批次只有一个center word，所以可以用独热向量$w_t$来表示。通过计算两者的乘积，我们就得到了当前想要的center word的向量$v_c$，$v_c \in R^{d*1}$。$v_c = w_t \cdot W$. 

图中的$W'$是context word矩阵，以行为单位存储每一个单词作为context word的向量表示，$W' \in R^{V*d}$。通过计算该矩阵和center word向量的内积我们可以得到一个中间产物$v_{tmp}$，$v_{tmp} = W' \cdot v_c$。对这个中间产物进行softmax，可以得到每一个词作为context word对应的概率，这个概率的向量表示标记为$p(x|c)$，是大小为$V$的向量$y_{pred}$，$p(x|c) = softmax(v_{tmp})$。我们希望在得到的向量$y_{pred}$中真正context word所对应的索引处的值（在上个模块例子中有4个context word）是大的，而其他索引处的值是小的。

$W$和$W'$都是模型训练过程中需要学习的。

<img src="https://i.postimg.cc/4NV9fKGV/w2v-theta.jpg" height="200">

之前提到每一个单词会有两个向量表示，即v (center word)和u (context word)，把这两个向量拼接起来（其实也可以相加）作为训练参数$\theta$，$\theta \in R^{2Vd}$。这里的$\theta$是一个非常长的向量，而不是一个矩阵。

## 连续词袋模型

英文是Continuous Bag of Words (CBOW)。Predict target word from bag-of-words context. 

### 目标函数Objective Function

Max the probability of center word given its context words. $\theta$ represents all variables we will optimize. The number of total words is T. Window size is m. 

$J'(\theta) = \prod_{t=1}^T \prod_{j = -m, j \ne 0}^m p(w_t|w_{t+j}; \theta)$

We use negative log likelihood to turn the objective function into a loss function. 

$J(\theta) = -\frac 1 {T} \sum_{t=1}^T \sum_{j = -m, j \ne 0}^m log p(w_{t}|w_{t+j})$

### 训练过程Training Process

<img src="https://i.postimg.cc/cJ4gBXnB/w2v-cbow.png" height="400">

训练过程和skig-gram非常类似。

When computing the hidden layer output, instead of directly copying the input vector of the input context word, the CBOW model takes the average of the vectors of the input context words, and use the product of the input→hidden weight matrix and the average vector as the output. 图中的$W$是context word矩阵，以列为单位存储每一个单词作为context word的向量表示，$W \in R^{d*V}$。如果在一个训练批次只考虑一个context word，可以用独热向量$x_t$来表示。通过计算两者的内积，我们就得到了当前想要的context word的向量$v_{context}$，$v_{context} \in R^{d*1}$。$v_{context} = W \cdot x_t$. 但是，在context包含多个词的时候，通常会采用这多个context word所对应向量的平均值作为输入。$v_{context} = \frac{1}{2m} \sum_{j=-m \\\\ j \ne 0}^m W \cdot x_j$.



# 近似训练

Word2vec在计算损失的时候使用到了softmax，所以需要考虑词典中的所有单词。对于上百万词的较大词典，计算梯度的开销会非常大。为了降低计算复杂度，研究人员提出了层序softmax和负采样两种近似训练方法。

## 层序SoftMax

英文是Hierarchical softmax。层序softmax将语言模型的输出softmax层编码为树形层次结构，其中每个叶子代表词典中一个单词，每个内部节点代表子节点的相对概率。

<img src="https://i.postimg.cc/rmsVKHt7/w2v-hs.png" height="200">

图中从根节点到$w_2$的示例路径被突出显示。$p^w$ 根节点到叶节点的路径。我们使用$l(w)$代表根结点到叶节点的路径（包括根节点和叶节点）上的结点数。例如，在图示中，$l(w_2)$是4。使用$n(w, j)$ 表示到叶节点$w$路径中的第$j$个节点，该节点的背景词向量是$u_{n(w, j)}$。$d_j^w \in \{0,1\}$ 是$p^w$上第$j$个节点的编码。$\theta_j^w$ 是$p^w$上第$j$个节点的向量。

在此模型中，没有单词的输出矢量表示。相当于是去掉了模型的隐藏层。原因是从hidden layer到output layer的矩阵运算太多了。

使用了哈夫曼树，时间复杂度就从$O(|V|)$降到了$O(log_2|V|)$。另外，由于哈夫曼树的特点，词频高的编码短，进一步加快了模型的训练过程。

### 损失函数Loss Function

#### with SG

条件概率可以近似表示为
$$
P(w_o|w_c) = \prod_{j=1}^{l(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o,j)) ]\!] \cdot \boldsymbol{u}_{n(w_o,j)}^\top \boldsymbol{v}_c\right),
$$
其中，$[\![ x]\!]$中的值如果为真，则表达式结果为1，否则结果为-1。

我们需要将中心词的向量和根节点到预测背景词路径上的非叶节点向量一一求内积。由于$\sigma(x)+\sigma(-x) = 1$，给定中心词$w_c$生成词典中任一词的条件概率之和为1这一条件也将满足。

#### with CBOW

## 负采样Negative Sampling

每次训练不需要更新所有负样本的权重，而只更新其中的k个。

Word2vec在词频基础上取了0.75次幂，减小词频之间差异过大所带来的影响，使得词频比较小的负样本也有机会被采到。

$$
weight(w) = count(w)^{0.75}/\sum_{i=1}^V count(i)^{0.75} \\\\
P(w) = U(w)^{0.75} / Z
$$

### 损失函数Loss Function

#### with SG

Our new objective function:

$log \sigma(u_{o}^T \cdot v_c) + \sum_{k=1}^K E_{j \sim P(w)} log \sigma(-u_j^T \cdot v_c)$.

Loss function:

$J_{neg}(o, v_c, U) = -log \sigma(u_o^Tv_c) - \sum_{k=1}^K log \sigma(-u_k^T \cdot v_c)$

This maximizes probability that real outside word appears, minimize probability that random words appear around center word. 



考虑一对单词(w,c)和上下文。使用$P(D = 1 | w,c,\theta)$表示(w,c)来自语料数据的概率。相应地，$P(D = 0 | w,c,\theta)$将是(w,c)不是来自语料数据的概率。

$$
P(D = 1 | w,c,\theta) = \sigma(u_o^Tv_c) = 1/(1+exp(-u_o^Tv_c))
$$
Now, we build a new objective function that tries to maximize the probability of a word and context being in the corpus data if it indeed is, and maximize the probability of a word and context not being in the corpus data if it indeed is not. We take a simple maximum likelihood approach of these two probabilities. (Here we take θ to be the parameters of the model, and in our case it is V and U.)
$$
\theta = argmax_\theta \prod_{(w,c) \in D} P(D=1| w,c,\theta) \prod_{(w,c) \in \widetilde D} P(D=0| w,c,\theta)  \\\\
= argmax_\theta \prod_{(w,c) \in D} P(D=1| w,c,\theta) \prod_{(w,c) \in \widetilde D} 1-P(D=1| w,c,\theta)  \\\\
= argmax_\theta \sum_{(w,c) \in D} log P(D=1| w,c,\theta) \sum_{(w,c) \in \widetilde D} log(1-P(D=1| w,c,\theta))  \\\\
= argmax_\theta \sum_{(w,c) \in D} log(1/(1+exp(-u_w^Tv_c)) \sum_{(w,c) \in \widetilde D} log(1-1/(1+exp(-u_w^Tv_c))  \\\\
= argmax_\theta \sum_{(w,c) \in D} log(1/(1+exp(-u_w^Tv_c)) \sum_{(w,c) \in \widetilde D} log(1/(1+exp(u_w^Tv_c))  \\\\
$$
$\widetilde D$ stands for a "false" corpus. For example, the unnatural sentences is one of such corpus. 

Our new objective function:

$log \sigma(u_{c-m+j}^T \cdot v_c) + \sum_{k=1}^K log \sigma(-\widetilde u_k^T \cdot v_c)$.

In the above formulation, {$\widetilde u_k$|k=1~K} are sampled from P(w). 



极大化正样本出现的概率，极小化负样本出现的概率。用sigmoid代替了softmax，相当于进行正负样本的二分类。

$E = -log \sigma(v'_{w_{pos}}h) - \sum_{w_j \in W_{neg}} log \sigma(-v'_{w_j}h)$

对于W'中v'进行求导

$v_{w_j}^{'(t+1)} = v_{w_j}^{'(t)} - \eta (\sigma(v_{w_j}^{'(t)T}h)-t_j)h$

#### with CBOW



# Doc2Vec

其实一个句子的向量表示可以由构成这个句子的词语的向量求加权平均值而得到。频率高的词权重稍微小一些。由这个方法得到的句子向量其实有非常不错的效果。虽然这种方法效果非常不错，但是有一个缺陷，就是忽略了单词之间的排列顺序对句子或文本信息造成的影响。比如，将一个句子的主语和宾语调换，那么意思会完全相反，但是根据这种方法得出来的向量却是相同的。

Doc2vec被用来解决这个问题，在使用向量表示段落或者文本的时候，考虑到了词序对于语意的影响。

**pros**

1. 非监督的学习方法，可以被应用于没有足够标签的训练数据

**cons**

1. missing new words (impossible to keep up to date)缺少新词

## 训练方法Training Algorithms

### Distributed Memory (PV-DM)

PV-DM is analogous to Word2Vec CBOW. The doc-vectors are obtained by training a neural network on the synthetic task of predicting a center word based an average of both context word-vectors and the full document’s doc-vector. It acts as a memory that remembers what is missing from the current context — or as the topic of the paragraph. 名字起得比较搞笑，PV-DM实际上对应的是word2vec中的CBOW模式。在给定上下文和文档向量的情况下预测单词的概率。

DM模型在训练时，首先将每个文档ID和语料库中的所有词初始化一个K维的向量，然后将文档向量和上下文词的向量输入模型，隐层将这些向量累加（或取均值、或直接拼接起来）得到中间向量，作为输出层softmax的输入。在一个文档的训练过程中，文档ID保持不变，共享着同一个文档向量，相当于在预测单词的概率时，都利用了整个文档的语义。

<img src="https://i.postimg.cc/t7QpvCmH/d2v-dm.png">

在这个图中，作者貌似只说了用前文的词去预测后文的词，比如在这个例子中"the cat sat"是"on"的前文。这个实际上具有一定的误导性。在原paper的损失函数仍然是包含了一个center word的前后文的。 

<img src="https://i.postimg.cc/pdG9GzQb/d2w-paper.png">



假设总共有N段文字，文字所映射到的向量维度是p。字典中词的数量是V，词所映射到的向量维度是q。那么模型总共有N\*p+V\*q个参数。

### Distributed Bag of Words (PV-DBOW)

PV-DBOW is analogous to Word2Vec SG. The doc-vectors are obtained by training a neural network on the synthetic task of predicting a target word just from the full document’s doc-vector.名字起得比较搞笑，PV-DM实际上对应的是word2vec中的SG模式。在每次迭代的时候，从文本中采样得到一个窗口，再从这个窗口中随机采样一个单词作为预测任务，让模型去预测，输入就是段落向量。

<img src="https://i.postimg.cc/XNT56fRn/d2v-dbow.png">

这种训练方式通常要比DM训练方式快很多，需要更少的储存空间，但是准确度不如DM高。

## 调包使用

### 参数

- **dm**: 0 = DBOW; 1 = DMPV. 模型的模式
- **vector_size**: Dimensionality of the feature vectors.
- **window**: The maximum distance between the current and predicted word within a sentence.
- **min_count**: Ignores all words with total frequency lower than this.
- **sample**: this is the sub-sampling threshold to downsample frequent words; 10e-5 is usually good for DBOW, and 10e-6 for DMPV. 
- **hs**: 1 turns on hierarchical sampling; this is rarely turned on as negative sampling is in general better
- **negative**: number of negative samples; 5 is a good value. 
- **dm_mean **(*optional*): If 0 , use the sum of the context word vectors. If 1, use the mean. Only applies when dm is used in non-concatenative mode.
- **dm_concat** (*optional*): If 1, use concatenation of context vectors rather than sum/average; Note concatenation results in a much-larger model, as the input is no longer the size of one (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words in the context strung together.
- **dbow_words** (*optional*): If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).

### 方法

```python
class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """
    A single document, made up of `words` (a list of unicode string tokens)
    and `tags` (a list of tokens). Tags may be one or more unicode string
    tokens, but typical practice (which will also be most memory-efficient) is
    for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from Word2Vec.
    """
```

很多人奇怪doc2vec作为一个非监督学习的方法，为什么会需要提供一个words tags的选项。通过看文档我们可以发现，实际上这个参数我们填写每个文档对应的唯一性标识就可以。当然，我们也可以传对应的标签进去，但是这个并不会妨碍doc2vec把文档当成标记过的数据。注意，必须要把标记当成列表传递。

### 示例

下面的案例是基于一个三分类的问题，使用的是dm模式。其中xtrain["ngram"]是已经分好词的预料。

```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def getVec(model, tagged_docs, epochs=20):
  sents = tagged_docs.values
  regressors = [model.infer_vector(doc.words, epochs=epochs) for doc in sents]
  return np.array(regressors)

def plotVec(ax, x, y, title="title"):
  scatter = ax.scatter(x[:, 0], x[:, 1], c=y, 
             cmap=matplotlib.colors.ListedColormap(["red", "blue", "yellow"]))
  ax.set_title(title)
  ax.legend(*scatter.legend_elements(), loc=0, title="Classes")

xtrain_tagged = xtrain.apply(
    lambda r: TaggedDocument(words=r["ngram"], tags=[r["Label"]]), axis=1
)

model_dm = Doc2Vec(dm=1, vector_size=30, negative=5, hs=0, min_count=2, sample=0)
model_dm.build_vocab(xtrain_tagged.values)
for epoch in range(10):
    sents = xtrain_tagged.values
    model_dm.train(sents, total_examples=len(sents), epochs=1)
    model_dm.alpha -= 0.002 
    model_dm.min_alpha = model_dm.alpha
xtrain_vec = getVec(model_dm, xtrain_tagged)
xtrain_tsne = TSNE(n_components=2, metric="cosine").fit_transform(xtrain_vec)
plotVec(ax1, xtrain_tsne, ytrain, title="training")
```





# Reference

- Stanford CS244n, [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html) 
- Stanford CS244d, [Deep Learning for NLP](https://cs224d.stanford.edu/lecture_notes/notes1.pdf)
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/), Chris McCormick
- [word2vec Parameter Learning Explained](https://arxiv.org/pdf/1411.2738.pdf), Xin Rong
- [doc2vec model](https://radimrehurek.com/gensim/models/doc2vec.html#module-gensim.models.doc2vec), gensim
- [Word2vec和Doc2vec原理理解并结合代码分析](https://blog.csdn.net/mpk_no1/article/details/72458003), mpk_no1
- [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053.pdf), Quoc Le & Tomas Mikolov
- [一篇通俗易懂的word2vec](https://zhuanlan.zhihu.com/p/35500923), susht, NLP Weekly
- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第10章

