# 语言建模

语言建模(language modeling)就是预测句子中下一个出现的单词的任务。更精确的说法是，给定已经出现的所有词，求下一个次出现的概率分布。
$$
P(w_{t+1}|w_1,...,w_t)
$$

## 语言模型

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

我们只考虑固定长度的窗口。文本仍然会被截断成为类似ngram一样的短句子，喂给模型。在每一个时间步，输出是字典所有词出现的概率。使用交叉熵(cross entropy)作为损失函数。

<img src="https://i.postimg.cc/vBY6Y0Xq/fixed-length-window-model.png" height=400>



**优点**

1. 改进了n元模型的稀疏的缺点
2. 不需要存储被观测到的n-gram

**缺点**

1. 窗口长度过小容易导致模型不准确；长度过大会导致参数过多
2. 学习进行时的参数未被共享（单词在不同时间步对应的学习参数并不相同）

### 循环神经网络

RNN可以用于语言建模，不过它的应用不仅限于此。

## 评判标准

### 困惑度

英文应该比中文更为人所熟知，perplexity。最标准最经典的指标。经过推导，我们发现就是交叉熵的指数。所以我们希望困惑度越小越好。
$$
perplexity = \prod_{t=1}^T (\frac 1 {P_{LM}(x^{(t+1)} | x^{(1)},...,x^{(t)}})^{1/T} \\\\
= \prod_{t=1}^T (1/\hat y_{x_{t+1}}^{(t)})^{1/T} \\\\
= exp(L(\theta))
$$

- 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
- 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
- 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。







# Reference

- [Natural Language Processing with Deep Learning (Stanford CS224N)](http://web.stanford.edu/class/cs224n/), 2019 winter

