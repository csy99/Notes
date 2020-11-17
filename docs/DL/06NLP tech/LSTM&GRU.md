# 长短期记忆

大名鼎鼎的Long short term memory (LSTM)。LSTM的提出，正是为了解决RNN的痛点，加强了捕捉时间序列中时间步距离较大的依赖关系的能力。

需要注意的是LSTM并没有保证梯度爆炸或者梯度消失的问题不会出现，它仅仅是为模型学习长依赖提供了一种更简便的方法(记忆细胞)。

我们再次熟悉一下对于各个变量的标记。假设激活函数为$\phi$，该层的隐藏单元个数是$h$，样本数为$n$，特征维度是$d$ (在此处就是字典大小)。所以数据可以表示成$X \in R^{n*d}$。

<img src="https://i.postimg.cc/gcVyjCNf/LSTM.png" height=300>

### 控制门

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。在时间步$t$，这3个门都是形状为$n*h$的矩阵。
$$
F^{(t)} = \sigma(H^{(t-1)}W_{hf} + X^{(t)}W_{xf} + b_f) \\\\
I^{(t)} = \sigma(H^{(t-1)}W_{hi} + X^{(t)}W_{xi} + b_i) \\\\
O^{(t)} = \sigma(H^{(t-1)}W_{ho} + X^{(t)}W_{x0} + b_o) \\\\
$$
其中$W \in R^{h*h}$是权重参数，$b \in R^{1*h}$是偏置参数。

遗忘门控制上一时间步细胞什么信息需要保留或丢弃。输入门控制当前候选记忆细胞哪一部分会被写入到记忆细胞。输出门控制记忆细胞中哪些信息会被写入到隐藏状态。

### 候选记忆细胞

英文叫做new cell content，代表会被写入到细胞的内容，常用标记是$\tilde C^{(t)}$。在计算上其实与控制门基本相同。唯一不同的是，它采用了tanh作为激活函数。
$$
\tilde C^{(t)} = \tanh(H^{(t-1)}W_{hc} + X^{(t)}W_{xc} + b_c) \\\\
$$

### 记忆细胞

英文是cell state。当前时间步记忆细胞$C^{(t)} \in R^{n*h}$的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，通过遗忘门和输入门来控制信息的流动。算式中$\circ$代表的是元素积(element-wise product)。
$$
C^{(t)} = F^{(t)} \circ C^{(t-1)} + I^{(t)} \circ \tilde C^{(t)}
$$
如果遗忘门一直近似1且输入门一直近似0，过去的记忆细胞将一直通过时间保存并传递至当前时间步。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

### 隐藏状态

英文是hidden state。通过输出门来控制从记忆细胞到隐藏状态的信息的流动。当输出门近似1时，记忆细胞信号将传递到隐藏状态供输出层使用；否则，记忆细胞信息只自己保留。
$$
H^{(t)} = O^{(t)} \circ \tanh(C^{(t)})
$$



# 门控循环单元

个人认为，门控循环神经网络（gated recurrent neural network, GRU）的提出更像是对LSTM的一种简化，而非改进。目前两种方法在不同的应用场景各有优秀的表现，并未出现其中一种方法全方位压制另一种方法的情况。但是，GRU相比较LSTM有一个巨大的优势，即参数少，计算更加快捷，极大提升了训练速率。

<img src="https://i.postimg.cc/BQNxcxWp/GRU.png" height=300>

### 控制门

GRU只有两个门。重置门(reset gate)和更新门(update gate)。重置门有助于捕捉时间序列里短期的依赖关系；更新门有助于捕捉时间序列里长期的依赖关系。
$$
R^{(t)} = \sigma(H^{(t-1)}W_{hr} + X^{(t)}W_{xr} + b_r) \\\\
Z^{(t)} = \sigma(H^{(t-1)}W_{hz} + X^{(t)}W_{xz} + b_z) \\\\
$$

### 候选隐藏状态

与LSTM的候选记忆细胞作用类似。重置门控制了上一时间步的隐藏状态如何流入当前时间步的候选隐藏状态。而上一时间步的隐藏状态可能包含了时间序列截至上一时间步的全部历史信息。因此，重置门可以用来丢弃与预测无关的历史信息。
$$
\tilde H^{(t)} = \tanh((R^{(t)} \circ H^{(t-1)})W_{hh} + X^{(t)}W_{xh} + b_h) \\\\
$$

### 隐藏状态

这个模块是GRU最有意思的小创新。当前步的隐藏状态既需要包含之前步的隐藏状态，也需要包含当前候选隐藏状态的信息。GRU直接使用更新门来控制两者的权重。
$$
H^{(t)} = (1-Z^{(t)}) \circ H^{(t-1)} + Z^{(t)} \circ \tilde H^{(t)}
$$










# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第6章
- [Natural Language Processing with Deep Learning (Stanford CS224N)](http://web.stanford.edu/class/cs224n/), 2019 winter

