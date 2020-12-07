# 衡量标准Evaluation

此处的衡量标准主要是针对单词向量而言。

### 内部标准Intrinsic

- 固定任务 (specific subtask)
- 运行时间 (time taken to compute)
- 能否帮助理解系统 (helps to understand the system)

### 外部标准Extrinsic

- 真实任务的评估 (evaluation on a real task)
- 计算正确率的用时 (time taken to compute accuracy)
- 不清楚是该子系统问题还是它与其他子系统的相互作用导致了问题的出现 (unclear if the subsystem is the problem or its interaction with other subsystems) 

### 内在词向量评估

英文是Intrinsic word vector evaluation。



# 引子

我们先从word2vec的跳字模型入手。考虑$w_i$为中心词时，将其所有背景词全部汇总并保留重复元素，记作多重集(multiset) $\mathcal C_i$。一个元素在多重集中的个数称为该元素的重数(multiplicity)。将多重集中$j$元素的重数记为$x_{ij}$，表示整个数据集中所有以$w_i$为中心词的背景窗口中词$w_j$的个数。损失函数可以表示如下。
$$
J = -\sum_{i \in V}\sum_{j \in V} x_{ij} log P(w_j|w_i)
$$
我们将数据集中所有以$w_i$为中心词的背景词的数量之和$|\mathcal C_i|$记为$x_i$，将以$w_i$为中心词生成背景词$w_j$的条件概率$x_{ij} / x_i$记作$p_{ij}$。我们可以将损失函数改写为
$$
J = -\sum_{i \in V} x_{i} \sum_{j \in V} p_{ij} log P(w_j|w_i)
$$
损失函数的后半部分计算的是以$w_i$为中心词的背景词条件概率分布$p_{ij}$和模型预测的条件概率分布的交叉熵，前半部分使用以$w_i$为中心词的背景词的数量之和对交叉熵进行加权。最小化损失函数会使模型预测的条件概率分布尽可能接近训练集的条件概率分布。

由于生僻词出现次数少，其权重较小。模型对生僻词的条件概率分布的预测会出现较大偏差。

类似Latent Semantic Analysis (LSA)等全局矩阵分解模型具有训练时间短和充分运用统计知识的优点，但是模型在词类比任务上表现较差，另外会给予常出现的词语非常大的权重。

而诸如word2vec和NNLM等上下文窗口模型能够学习到除词语相似度之外的一些复杂规律，但是可扩展性较弱（不太能够应对非常大的训练文本），而且对统计知识的利用不足（忽视一些重复出现的词句现象）。

下面介绍全局向量的词嵌入模型，结合了上述两类模型的优点。



# 全局向量的词嵌入

英文是Gloabl Vectors for Word Representation (GloVe)。

### 损失函数

采用平方损失取代交叉熵，使用非概率分布的变量$p’_{ij} = x_{ij}$和$q’_{ij} = exp(u_j^Tv_i)$，并对他们取对数。
$$
J = (log p’_{ij} - log q’_{ij})^2 = (u_j^Tv_i - log(x_{ij}))^2
$$
为每一个词$w_i$增加了两个标量模型参数：中心词偏差项$b_i$和背景词偏差项$c_j$。将每个损失项的权重替换成函数$h(x_{ij})$，值域在[0,1]之间，单调递增。
$$
\sum_{i \in V} \sum_{j \in V} h(x_{ij}) (u_j^Tv_i+b_i+c_j- log(x_{ij}))^2
$$
其中权重函数$h(x)$的建议选择如下：如果$x < t$，令$h(x) = (x/t)^{\alpha}$，如$\alpha = 0.75$。否则令$h(x) = 1$。此处$t$只是一个人为设定的最大界限。因为$h(0) = 0$，所以如果两个词并未相邻，则不会有任何损失。每个时间步采样小批量非零$x_{ij}$(可以预先基于整个数据集计算得到)，包含了数据集的全局统计信息。

不同于word2vec中拟合的是非对称的条件概率$P(w_j|w_i)$，GloVe模型拟合的是对称的$x_{ij}$。因此，任意词的中心词向量和背景词向量在GloVe模型中是等价的。但由于初始化值的不同，同一个词最终学习到的两组词向量可能不同。当学习得到所有词向量以后，GloVe模型使用中心词向量与背景词向量之和作为该词的最终词向量。

### 条件概率比值

作为源于某大型语料库的真实例子，以下列举了两组分别以“ice”（冰）和“steam”（蒸汽）为中心词的条件概率以及它们之间的比值。

<img src="https://i.postimg.cc/tTVqKcgY/glove-ratio.png" height=150>

对于与“ice”相关而与“steam”不相关的词$w_k$，如“solid”，我们期望条件概率比值较大，如上表最后一行中的值8.9；

对于与“ice”不相关而与“steam”相关的词$w_k$，如“gas”，我们期望条件概率比值较小，如上表最后一行中的值0.085；

对于与“ice”和“steam”都相关的词或都不相关的词$w_k$，如“water”和“fashion”，我们期望条件概率比值接近1，如上表最后一行中的值1.36和0.96；

我们可以对任意三个不同的词构造条件概率比值。
$$
f(u_j, u_k, v_i) \approx \frac{p_{ij}} {p_{ik}}
$$
函数$f$可能的设计并不唯一，我们只考虑其中一种合理的可能性。由于函数输出是一个标量，我们可以将函数限制为一个标量函数。
$$
f(u_j, u_k, v_i) = f((u_j-u_k)^T v_i)
$$
值得注意的是单词共现矩阵中，背景词和中心词应该是可以互换的。我们需要使得$u \Leftrightarrow v$（单词作为背景词与作为中心词的词向量等价），还要使得单词共现矩阵$X = X^T$。而显然上式并不满足这两个条件。

我们利用函数在群$(R,+)$和$R_{>0}, \times$同态知识改写上式。
$$
f((u_j-u_k)^T v_i) = \frac{f(u_j^T v_i)}{f(u_k^T v_i)} \\\\
f(u_j^T v_i) = P(w_j|w_i) = \frac {x_{ij}}{x_i}
$$
因此，一种可能是$f(x) = exp(x)$。于是，
$$
f(u_j, u_k, v_i) = \frac{exp(u_j^T v_i)} {exp(u_k^T v_i)} \approx \frac{p_{ij}} {p_{ik}}
$$
满足约等号的一种可能是$exp(u_j^T v_i) \approx \alpha p_{ij}$，考虑到$p_{ij} = x_{ij}/x_i$，取对数后$u_j^T v_i \approx log(\alpha) + log(x_{ij}) - log(x_i)$。我们使用额外的偏差项来拟合$-log(\alpha) + log(x_i)$，可以得到
$$
u_j^T v_i - log(\alpha) + log(x_i) \approx log(x_{ij})
$$
对上式左右两边取平方误差并加权，同样可以得到GloVe模型的损失函数。

### 时间复杂度

模型的复杂度主要受限于单词共现矩阵$X$中的非零项，一个较为宽松的上限是$O(|V|)^2$。这看起来像是废话。我们再尝试缩小这个上限。

我们假设一组词$w_i$和$w_j$同时出现的几率可以被模拟为该词频率等级的幂律函数$r_{ij}$。
$$
X_{ij} = \frac{k} {(r_{ij})^{\alpha}}
$$
我们知道，训练文本中的单词总数与共现矩阵中的每一项之和成正比。我们使用广义调和数重新改写了总合。
$$
|C| \sim \sum_{ij} X_{ij} = \sum_{r=1}^{|X|} \frac k {r^\alpha} = kH_{|X|,\alpha}
$$
这个和的上界是$|X|$，同时是最大频率等级，也是矩阵中的非零元素。也是当$X_{ij} \ge 1$时$r$可取的最大值。根据此，我们再将上式进行改写。
$$
|C| \sim  |X|^\alpha H_{|X|,\alpha}
$$
同时，我们已知广义调和数的计算($s>0$且$s\ne1$)，其中$\mathcal L(s)$是黎曼Zeta函数。
$$
H_{x,s} = \frac{x^{1-s}} {1-s} + \mathcal L(s) + \mathcal O(x^{-s})
$$
再对上式进行化简
$$
|C| \sim  \frac{|X|}{1-\alpha} \mathcal L(\alpha) |X|^\alpha + \mathcal O(1)
$$
当$X$非常大时
$$
|X| = \mathcal O(|C|), \alpha < 1 \\\\
|X| = \mathcal O(|C|^{1/\alpha }), \alpha > 1 \\\\
$$
经过试验，在原paper中得到$\alpha=1.25$，所以$|X| = \mathcal O(|C|^{0.8})$。所以这个比类似word2vec等窗口模型的复杂度$\mathcal O(|C|)$是要稍微好一些的。



# 应用

预训练的GloVe模型的命名规范大致是“模型.（数据集.）数据集词数.词向量维度.txt”。

我们平时可以使用基于维基百科子集预训练的50维GloVe词向量。其中含有40万个词和1个特殊的未知词符号。

### 近义词

```python
from mxnet import nd

def knn(W, x, k):
	# 添加的1e-9是为了数值稳定性
    cos = np.dot(W, x.reshape((-1,))) / (np.sqrt((nd.sum(W * W, axis=1) + 1e-9)) * np.sqrt(np.sum(x * x)))
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec,
                    embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))
```

### 类比词

由近义词延伸出来的类比词。对于类比关系中的4个词$a:b :: c:d$， 第一组词$a$和$b$在某个维度上不同，而且这种差异可以在第二组词$c$和$d$上体现。给定前3个词 $a$ 、$b$和$c$ ，求$d$。设词$w$的词向量为vec(w) 。求类比词的思路是，搜索与vec(c)+vec(b)−vec(a)的结果向量最相似的词向量。需要注意的是，如果搜索结果中出现了输入的单词，我们需要丢弃。
$$
d = argmax_i \frac{(x_b-x_a+x_c)^T x_i} {||x_b-x_a+x_c||}
$$
下图展示了不同词之间的对应关系，词嵌入的差异主要是由于性别造成的。

<img src="https://i.postimg.cc/hthrqXFp/glove-visualization.png" height=400>

代码实现如下。

```python
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]
```

例如

```python
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

可以得到答案是japan。相当于$Beijing:China :: Tokyo:Japan$。

**缺点**

1. 单词之间具有一定的关联，但未必是线性关联



# 对一词多义的思考

大部分单词都有很多个意思，特别是那些存在已久的单词和常见的单词。如何解决一词多义的问题是词嵌入的一大难题。

### 多单词原形训练

在GloVe中，我们可以创造多个单词原形进行学习，将多个单词原形放到不同聚类当中进行重新训练。例如，bank既可以表示银行，也可以表示河边、岸边意思。下图bank(1)(2)分别代表了这个意思。

<img src="https://i.postimg.cc/ZRGdXLth/multi-meaning.png" height=400>

### 一词多义的线性组合

例如word2vec之类的标准单词嵌入，使用单词的不同含义的加权线性叠加。我们可以设置权重为单词不同意思出现的概率。
$$
vec_{word} = \alpha_1 vec_{word_1} +  \alpha_2 vec_{word_2} + \alpha_3 vec_{word_3} \\\\
\alpha_1 = \frac{f_1} {f_1+f_2+f_3}
$$








# Reference

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington, Richard Socher, Christopher D. Manning. 2014. 
- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第10章