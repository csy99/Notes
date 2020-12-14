# 时序数据的采样

### 随机采样

每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

### 相邻采样

相邻的两个随机小批量在原始序列上的位置相毗邻。可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量的输入，并如此循环下去。这对实现循环神经网络造成了两方面影响：一方面， 在训练模型时，我们只需在每一个迭代周期开始时初始化隐藏状态；另一方面，当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。 为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图中分离出来。



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

我们首先考虑不含隐藏状态的隐藏层。假设激活函数为$\phi$，该层的隐藏单元个数是$h$，样本数为$n$，特征维度是$d$ (在此处就是字典大小)。所以数据可以表示成$X \in R^{n*d}$，该层权重是$W \in R^{d*h}$。

该隐藏层之后的输出层维度是$q$。
$$
H = \phi(XW_{xh} + b_h) \\\\
O = HW_{hq} + b_q
$$
现在开始考虑输入数据存在时间相关性的情况。请不要将时间步和矩阵转置混淆，所有出现在变量上方位置的时间步均有括号覆盖。数据可以表示成$X^{(t)} \in R^{n*d}$，而该时间步的隐藏变量可以表示成$H^{(t)} \in R^{n*h}$。我们需要引入一个新的权重参数描述当前时间步如何使用上一时间步的隐藏变量，记作$W_{hh} \in R^{h*h}$。
$$
H^{(t)} = \phi(H^{(t-1)}W_{hh} + X^{(t)}W_{xh} + b_h) \\\\
O^{(t)} = H^{(t)} W_{hq} + b_q
$$
下面我们看一个字符级循环神经网络的例子。设小批量中样本数为1，文本序列为“想要有直升机”。下图演示了如何使用循环神经网络基于当前和过去的字符来预测下一个字符。在训练时，我们对每个时间步的输出层输出使用softmax运算，然后使用交叉熵损失函数来计算它与标签的误差。

<img src="https://i.postimg.cc/nzWH714r/rnn1.png" height=200>

使用循环神经网络来预测一段文字的下一个词，输出个数是字典中不同词的个数。激活函数$\phi$一般会设置为tanh或relu。

### 损失函数

对于每一步都预测一个词。在第t步的损失如下，
$$
L^{(t)}(\theta) = -\sum_{w \in V} y_w^{(t)} log \hat y_w^{(t)} = -log \hat y_{vec[x_{t+1}]}^{(t)}
$$
我们可以将时间步$t$的损失记为$l(\hat y^{(t)}, y^{(t)})$。对于数据整体(时间步数为$T$)的损失就是对每一步的损失取平均值。我们称这个算式为有关给定时间步的数据样本的目标函数。
$$
L = \frac 1 T \sum_{t=1}^T l(\hat y^{(t)}, y^{(t)})
$$

### 反向传播

英文是back propagation。

下面这个结论仍然是根据链式法则得到的。中间第二项是1。
$$
\frac {\partial L^{(T)}} {\partial W_h} = \sum_{t=1}^T \frac {\partial L^{(T)}} {\partial W_h |_{(t)}} \frac {\partial W_h |_{(t)}} {\partial W_h}= \sum_{t=1}^T \frac {\partial L^{(t)}} {\partial W_h |_{(t)}}
$$
难点是我们如何计算这个算式。我们只能使用基于时间的反向传播算法(backpropagate through time， BPTT)。

<img src="https://i.postimg.cc/027Y6LnT/rnn-bptt.png" height=200>

上图展示时间步数为3的循环神经网络模型计算中的依赖关系。方框代表变量（无阴影）或参数（有阴影），圆圈代表运算符。

目标函数有关各时间步输出层变量的梯度比较容易计算。
$$
\frac {\partial L} {\partial \hat y^{(t)}} = \frac {\partial l(\hat y^{(t)}, y^{(t)})} {T*\partial \hat y^{(t)}}
$$
我们再计算目标函数有关参数$W_{hq}$的梯度，该梯度取决于所有时间步的输出变量。
$$
\frac {\partial L} {\partial W_{hq}} = \sum_{t=1}^T \frac {\partial L} {\partial \hat y^{(t)}}*\frac {\partial \hat y^{(t)}} {\partial W_{hq}} = \sum_{t=1}^T \frac {\partial L} {\partial \hat y^{(t)}} H^{(t)}
$$
接下来我们计算隐藏状态的依赖。$L$只通过最后一个预测依赖最终时间步的隐藏状态。
$$
\frac {\partial L} {\partial H^{(T)}} = \frac {\partial L} {\partial \hat y^{(T)}}*\frac{\partial \hat y^{(T)}} {\partial H^{(T)}} = W_{hq}^T \frac {\partial L} {\partial \hat y^{(T)}}
$$
不过，对于之前的时间步$t < T$，$L$会通过$H^{(t+1)}$和$\hat y^{(t)}$依赖$H^{(t)}$。目标函数关于时间步$t$的隐藏状态的梯度需要按照时间步从大到小一次计算。
$$
\frac {\partial L} {\partial H^{(t)}} = \frac {\partial L} {\partial H^{(t+1)}}*\frac{\partial H^{(t+1)}} {\partial H^{(t)}} + \frac {\partial L} {\partial \hat y^{(t)}}*\frac{\partial \hat y^{(t)}} {\partial H^{(t)}} \\\\
= W_{hh}^T \frac {\partial L} {\partial H^{(t+1)}} + W_{hq}^T \frac {\partial L} {\partial \hat y^{(t)}}
$$
我们可以将公式递归展开，对于任意时间步，得到目标函数有关隐藏状态的梯度的通项公式如下，
$$
\frac {\partial L} {\partial H^{(t)}} = \sum_{i=t}^T (W_{hh}^T)^{(T-i)} W_{hq}^T \frac {\partial L} {\partial \hat y^{(T+t-i)}}
$$
我们可以看到，$W_{hh}$在公式中有一个指数项，当$T-i$较大时，容易出现梯度爆炸和梯度消失的问题。而且，这个问题也会严重影响其他包含该项的梯度。
$$
\frac {\partial L} {\partial W_{xh}} = \sum_{t=1}^T \frac {\partial L} {\partial H^{(t)}}*x_t^T \\\\
\frac {\partial L} {\partial W_{hh}} = \sum_{t=1}^T \frac {\partial L} {\partial H^{(t)}}* (H^{(t-1)})^T \\\\
$$

### 应用场景

语言建模，语句情感分析，词性标注。



## 参数更新

### 梯度消失

<img src="https://i.postimg.cc/Hs15C6mB/vanishing-gradient.png" height=300>

如果这些取导得到的值都非常小，那么在反向传播之后，距离输出层较远的隐层的参数更新会非常缓慢。

上面我们已经推导过一遍。为了计算简便，看起来更加直观，我们假设激活函数是单位函数$\phi(x) = x$。
$$
\frac{\partial h^{(t)}}{\partial h^{(t-1)}} \\\\
= \frac {\partial \phi(W_{x}x^{(t)} +  W_{h} h^{(t-1)} + b)} {\partial h^{(t-1)}} \\\\
= W_h
$$
假设我们正在计算第$i$步的损失的梯度，那么对于之前的第$j$步有如下算式。

$$
\frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} \\\\
= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{j<t\le i} \frac{\partial h^{(t)}} {\partial h^{(t-1)}} \\\\
= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{j<t\le i} W_h \\\\
=  \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_h^{i-j}
$$
我们注意到，如果$W_h$非常小的话，那么$W_h^{i-j}$则是成指数倍减小。

*梯度消失到底有什么坏处呢？*

坏处就是导致RNN在实际应用中难以产生理论上所拥有的长期记忆。因为在经过若干步之后，前面的梯度基本不受到当前步的影响了。出现这种情况的原因既有可能是若干步之前的数据本身就对当前步没有影响，即在数据中长依赖关系本身就不存在；也有可能是参数没有被及时更新，从而导致这种长依赖无法被捕捉到。我们没有办法判断是这两种原因中的哪一种。

在文本处理中，RNN更擅长处理语序新近度(sequential recency)而非语法新近度(syntactic recency)，从而更有可能导致在填词的时候判断错误。语法新近度指当前词是用来修饰之前出现过的哪一个词的，而语序新近度是指当前词的前一个词(距离最近)。

### 梯度爆炸

梯度爆炸出现的原因与梯度消失非常类似，只不过是梯度过大。坏处是容易导致

#### 梯度剪裁

英文gradient clipping应该更为大家所熟知。如果梯度的模大于一个临界值，那么需要在保留方向的前提下减小其模长。

```python
norm = np.linalg.norm(g)
if norm > threshold:
    g = threshold/norm * g
```



# 双向循环神经网络

英文是(Bidirectional RNN)。之前介绍的循环神经网络模型都是假设当前时间步是由前面的较早时间步的序列决定的，因此它们都将信息通过隐藏状态从前往后传递。有时候，当前时间步也可能由后面时间步决定。双向循环神经网络在机器翻译和情感分析中应用广泛，在语言建模中没有用武之地(只给出前文)。所以，使用双向循环神经网络的前提条件就是有完整的序列。
$$
\stackrel{\rightarrow}{H}^{(t)} = RNN_{FW}(\stackrel{\rightarrow}{H}^{(t-1)}, X^{(t)}) \\\\
\stackrel{\leftarrow}{H}^{(t)} = RNN_{BW}(\stackrel{\leftarrow}{H}^{(t+1)}, X^{(t)}) \\\\
{H}^{(t)} = [\stackrel{\rightarrow}{H}^{(t)}; \stackrel{\leftarrow}{H}^{(t)}]
$$



# 深度循环神经网络

英文是multi-layer RNN。深度循环神经网络能够学习到更复杂的信息。较浅的RNN应计算较低级别的特征，较深的RNN应计算较抽象的特征。

隐藏状态的信息不断传递至当前层的下一时间步和当前时间步的下一层。

<img src="https://i.postimg.cc/g01jH0kN/mutli-layer-rnn-arch.png" height=200>

第一隐藏层的隐藏状态和之前计算完全相同。
$$
H^{(t,l)} = \phi(H^{(t-1,l)}W_{hh}^{(l)} + X^{(t)}W_{xh}^{(l)} + b_h^{(l)}) \\\\
$$
对于层数较深的隐藏层，表达式变成
$$
H^{(t,l)} = \phi(H^{(t-1,l)}W_{hh}^{(l)} + H^{(t,l-1)}W_{xh}^{(l)} + b_h^{(l)}) \\\\
$$
输出层的输出仍然只基于第$L$层的隐藏状态。
$$
O^{(t)} = H^{(t,L)} W_{hq} + b_q
$$



# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第6章
- [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/), Stanford University, CS224N, 2019 winter

