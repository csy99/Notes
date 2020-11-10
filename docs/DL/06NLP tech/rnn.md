# 语言建模

语言建模就是预测句子中下一个出现的单词的任务。更精确的说法是，给定已经出现的所有词，求下一个次出现的概率分布。
$$
P(w_{t+1}|w_1,...,w_t)
$$

### 传统模型

假设当前词与之前所有的词有关。
$$
P(w_1, w_2, \ ...\ , w_T) = \prod_{t=1}^T P(w_t | w_{1}, \ ...\ , w_{t-1})
$$
**缺点**

1. 参数过多，计算复杂

### n元语法模型

n元的英文是N-Grams (n-gram)。 句子中连续的n个单词。

这里的马尔可夫假设是指一个词的出现只与前面n个词相关，即n阶马尔可夫链（Markov chain of order n）。
$$
P(w_1, w_2, \ ...\ , w_T) = \prod_{t=1}^T P(w_t | w_{t-n+1}, \ ...\ , w_{t-1})
$$
我们采用MLE估计。模型的实质就是预测为训练集中最常出现的组合。下面公式中context即为被预测词前面出现的n-1个单词。
$$
P(w|context) = \frac {count(context + w)} {count(context)}
$$
我们可以采用smoothing或者back-off（即使用n-1 gram模型，例如将4-gram降级为3-gram）的方法解决训练集中从未出现该组合的问题。

**优点**

1. 简化了语言模型

**缺点**

1. 在n较小时，n元语法往往并不准确
2. 当n较大时，需要大量的计算资源和内存资源；会加剧稀疏的问题

### 固定窗口神经语言模型

我们只考虑固定长度的窗口。

<img src="https://i.postimg.cc/vBY6Y0Xq/fixed-length-window-model.png" height=400>



**优点**

1. 改进了n元模型的稀疏的缺点
2. 不需要存储被观测到的n-gram

**缺点**

1. 窗口长度过小容易导致模型不准确；长度过大会导致参数过多
2. 学习进行时的参数未被共享（不同单词对应的学习参数并不相同）

### 循环神经网络

RNN可以用于语言建模，不过它的应用不仅限于此。



# 循环神经网络

Recurrent Neural Network (RNN). 下面我们讨论的架构通常被称为vanilla RNN，即最经典的RNN。

<img src="https://i.postimg.cc/JzStZDG3/rnn-arch.png" height=330>

**优点**

1. 能够解决记忆的问题，可以处理长序列
2. 通过隐藏状态来存储之前时间步的信息，参数数量不随时间步的增加而增长
3. 每个时间步的学习参数都相同

**缺点**

1. 计算非常缓慢（不能并行运行）
2. 实际上只能保存较短暂的内容，序列较长时，最前面的信息产生的作用会非常微弱
3. 梯度消失的问题

我们首先考虑不含隐藏状态的隐藏层。假设激活函数为$\phi$，该层的隐藏单元个数是$h$，样本数为$n$，特征维度是$d$。所以数据可以表示成$X \in R^{n*d}$，该层权重是$W \in R^{d*h}$。

该隐藏层之后的输出层维度是$q$。
$$
H = \phi(XW_{xh} + b_h) \\\\
O = HW_{hq} + b_q
$$
现在开始考虑输入数据存在时间相关性的情况。数据可以表示成$X_t \in R^{n*d}$，而该时间步的隐藏变量可以表示成$H_t \in R^{n*h}$。我们需要引入一个新的权重参数描述当前时间步如何使用上一时间步的隐藏变量，记作$W_{hh} \in R^{h*h}$。
$$
H_t = \phi(X_tW_{xh} + H_{t-1}W_{hh} + b_h) \\\\
O_t = H_t W_{hq} + b_q
$$
下面我们看一个字符级循环神经网络的例子。设小批量中样本数为1，文本序列为“想要有直升机”。下图演示了如何使用循环神经网络基于当前和过去的字符来预测下一个字符。在训练时，我们对每个时间步的输出层输出使用softmax运算，然后使用交叉熵损失函数来计算它与标签的误差。

<img src="https://i.postimg.cc/nzWH714r/rnn1.png" height=200>

使用循环神经网络来预测一段文字的下一个词，输出个数是字典中不同词的个数。激活函数$\phi$一般会设置为tanh或relu。

### 损失函数

对于每一步都预测一个词。在第t步的损失如下，
$$
L^{(t)}(\theta) = -\sum_{w \in V} y_t log \hat y_t = -vec[log \hat y_{t}]
$$
对于数据整体的损失就是对每一步的损失取平均值。

### 应用场景

语言建模，语句情感分析，词性标注。





# 时序数据的采样

### 随机采样

每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

### 相邻采样

相邻的两个随机小批量在原始序列上的位置相毗邻。可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量的输入，并如此循环下去。这对实现循环神经网络造成了两方面影响：一方面， 在训练模型时，我们只需在每一个迭代周期开始时初始化隐藏状态；另一方面，当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。 为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图中分离出来。







# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第6章
- [Natural Language Processing with Deep Learning (Stanford CS224N)](http://web.stanford.edu/class/cs224n/), 2019 winter
