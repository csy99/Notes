# 序列到序列网络

序列到序列网络(Sequence to Sequence network)，也叫做seq2seq网络, 又或者是编码器解码器网络(Encoder Decoder network)， 是一个由两个称为编码器解码器的RNN组成的模型。在这里进行介绍的作用是确定变量的名称，为接下来讲注意力机制做铺垫。

自编码器也是seq2seq模型中的一种。在自编码器中，解码器的工作是将编码器产生的向量还原成为原序列。经过压缩之后不可避免的会出现信息的损失，我们需要尽量将这种损失降低（方法是设置合理的中间向量大小和经过多次训练迭代的编码-解码器）。相关理论本身在这里讲得不是很清楚，有兴趣了解更多的同学可以移步数学专题中的[信息论](https://blog.csdn.net/qq_40136685/article/details/110849213)。

## 编码器

把一个不定长的输入序列变换成一个定长的背景变量$c$，并在该背景变量中编码输入序列信息。编码器可以使用循环神经网络。

在时间步$t$，循环神经网络将输入的特征向量$x_t$和上个时间步的隐藏状态$h_{t−1}$变换为当前时间步的隐藏状态$h_t$。
$$
h_t = f(x_t, h_{t-1})
$$
接下来，编码器通过自定义函数$q$将各个时间步的隐藏状态变换为背景变量
$$
c=q(h_1,...,h_T)
$$
例如，我们可以将背景变量设置成为输入序列最终时间步的隐藏状态$h_T$。

以上描述的编码器是一个单向的循环神经网络，每个时间步的隐藏状态只取决于该时间步及之前的输入子序列。我们也可以使用双向循环神经网络构造编码器。在这种情况下，编码器每个时间步的隐藏状态同时取决于该时间步之前和之后的子序列（包括当前时间步的输入），并编码了整个序列的信息。

## 解码器

对每个时间步$t′$，解码器输出$y_{t′}$的条件概率将基于之前的输出序列$y_1,...,y_{t′−1}$和背景变量$c$（所有时间步共用），即$P(y_{t′}∣y_1,...,y_{t′−1},c)$。同时还要考虑上一时间步的隐藏状态$s_{t′−1}$。
$$
s_{t'} = g(y_{t′−1}, c, s_{t'-1})
$$



# 注意力机制

### 动机

普通的seq2seq模型到底有什么问题？我们再来回顾一下输出流程。

<img src="https://i.postimg.cc/TwTc8D0T/seq2seq-bottleneck.png" height=300>

我们发现，解码器过分依赖于编码器在最后时间步生成的向量。这个向量需要包括源序列所有的信息，否则解码器将难以产生准确的输出。这个要求对于最后时间步生成的向量而言实在太高了。这个问题被称为信息瓶颈(information bottleneck)。

实际上，解码器在生成输出序列中的每一个词时可能只需利用输入序列某一部分的信息。为了让模型在不同时间步能够根据信息的有用程度分配权重，我们引入注意力机制。注意力机制的核心思想是解码器在每个时间步上使用直连接通向编码器以便其专注于源序列的部分内容。

### 过程

我们先通过图片形式展示过程。

<img src="https://i.postimg.cc/y81MXv2T/attention-process1.png" height=400>

在注意力得分(attention scores)那个区域的小圆圈代表向量点乘，使用softmax函数将注意力得分转换成注意力分布(attention distribution)。我们可以看到，在第一步，大部分注意力被分配给了源序列的第一个单词。通过加权相加的方法得到了编码器的一个隐状态。将编码器产生的隐状态和解码器的隐状态进行叠加，就可以产生解码器在该时间步的输出。

<img src="https://i.postimg.cc/BZy5C6y7/attention-process2.png" height=350>

有时，我们将注意力在上一步的输出当成输入，与当前的解码器隐状态一起进行训练，产生当前时间步的输出。例如这里的"he"就是上一步解码器的输出。

从另一个角度理解，对于普通的解码器，我们每一个时间步可以使用相同的背景变量。而在引入注意力机制之后，我们必须给每一个时间步不同的背景变量。
$$
s_{t'} = g(y_{t′−1}, c_{t'}, s_{t'-1})
$$
<img src="https://i.postimg.cc/ryHV9whR/attention.png" height=200>

我们用$h_1, ..., h_N \in R^{h}$来表示编码器的隐状态(hidden states)，用$s_{t'} \in R^{h}$来表示解码器在时间步$t'$的隐状态。我们可以计算注意力得分(attention scores)
$$
e_{t'} = [s_{t'}^Th_i,...,s_{t'}^Th_N] \in R^N
$$
使用softmax得到在时间步$t'$的注意力分布(attention distribution)
$$
\alpha_{t'} = softmax(e_{t'}) \in R^N
$$
利用分布计算加权总合得到注意力输出向量(attention output) $a$，有时候也叫做上下文向量(context vector) $c$，
$$
a_{t'} = c_{t'} = \sum_{i=1}^N \alpha_{i,t'}h_i \in R^{h}
$$
最后，我们将该向量与解码器当前的隐状态进行拼接
$$
[a_{t'}; s_{t'}] \in R^{2h}
$$

### 拓展

**优点**

1. 解决了信息瓶颈的问题
2. 缓解了梯度消失的问题
3. 在一定程度上提供了可解释性
4. 解决了软对齐问题（模型自己学习的，不需要人工调控）

注意力机制不光应用在机器翻译，还在很多其他领域有用武之地。

注意力机制的更通用的定义：给定一组值向量(values)和一个查询向量(query)，注意力是一种根据查询向量对值向量计算加权和的技术。我们有时候会说查询关注这些值(query attends to the values)。例如，在seq2seq+attention模型中，每一个解码器的隐状态(query)都会关注所有编码器的隐状态(values)。

我们用$h_1, ..., h_N \in R^{d_1}$来表示values，用$s \in R^{d_2}$来表示query。注意力的计算过程如下：

1. 计算注意力得分(attention scores)，$e \in R^N$
2. 进行softmax操作，将得分转换成注意力分布，$\alpha = softmax(e) \in R^N$
3. 利用分布计算加权总合得到输出向量(attention output) $a$，有时候也叫做上下文向量(context vector)，$a = \sum_{i=1}^N \alpha_ih_i \in R^{d_1}$

#### 计算注意力得分的方法

计算注意力得分的方式有很多种。

**点乘注意力**

英文是basic dot-product attention。使用前提是$d_1 = d_2$。
$$
e_i = s^Th_i
$$
**乘法注意力**

英文是multiplicative attention。权重矩阵$W \in R^{d_2*d_1}$。
$$
e_i = s^TWh_i
$$
**加法注意力**

英文是additive attention。权重矩阵$W_1 \in R^{d_3*d_1}$，$W_2 \in R^{d_2*d_1}$，权重向量$v \in R^{d_3}$。我们用$d_3$表示注意力维度(attention dimensionality)，这是一个超参数。
$$
e_i = v^Ttanh(W_1h_i + W_2s)
$$

### 矢量化计算

可以对注意力机制采用更高效的矢量化计算。我们再引入与值项一一对应的键项(key) $K$。

我们考虑编码器和解码器的隐藏单元个数均为$h$的情况。假设我们希望根据解码器单个隐藏状态$s_{t′−1} \in R^h$和编码器所有隐藏状态$[h_1, ..., h_N] \in R^{h}$来计算背景向量$c_{t′} \in R^h$。 我们可以将查询项矩阵$Q \in R^{1×h}$设为$s_{t′−1}^T$，并令键项矩阵$K \in R^{T×h}$和值项矩阵$V \in R^{T×h}$相同且第$t$行均为$h_t^T$。此时，我们只需要通过矢量化计算
$$
softmax(QK^T)V
$$
即可算出转置后的背景向量$c_{t′}^T$。当查询项矩阵$Q$的行数为$n$时，上式将得到$n$行的输出矩阵。输出矩阵与查询项矩阵在相同行上一一对应。










# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第10章

- [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html), Stanford CS244n, 2019 winter,  Chris Manning

  
