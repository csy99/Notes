# 高斯判别分析

全称是Gaussian Discriminant Analysis (GDA)。大家不要被名字所误导，这是一种概率生成模型。

## 原理

对联合概率进行建模，我们假设$y \sim Bernoulli(\Phi)$，且$x|y=1 \sim N(\mu_1, \Sigma)$，$x|y=0 \sim N(\mu_0, \Sigma)$

### 输入

训练集数据$D = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R^K$，二分类$y_i \in  \{-1, +1\}$

$X_i, \mu_i, \Sigma_i$ 分别表示第i类样例的集合、均值向量、协方差矩阵。

### 输出

各分类的概率。如果是二分类就是{+1, -1}的概率。

### 损失函数

令$\theta = (\mu_0, \mu_1, \sigma, \Phi)$。计算似然
$$
L(\theta) = \sum_{i=1}^M log[P(x_i|y_i)P(y_i)] \\\\
= \sum_{i=1}^M log P(x_i|y_i) + log P(y_i) \\\\
= \sum_{i=1}^M log[N(\mu_1|\Sigma)^{y_i} N(\mu_0|\Sigma)^{1-y_i}] + log [\Phi^{y_i}(1-\Phi)^{1-y_i}] \\\\
= \sum_{i=1}^M log[N(\mu_1|\Sigma)^{y_i}] + log[N(\mu_0|\Sigma)^{1-y_i}] + log [\Phi^{y_i}(1-\Phi)^{1-y_i}] \\\\
$$
对$\Phi$进行求偏导，
$$
\frac{\partial L(\theta)}{\partial \Phi} = \sum_{i=1}^M \frac{y_i} {\Phi} + \frac {1-y_i} {1-\Phi} = 0\\\\
\sum_{i=1}^M  {\Phi}(1-y_i) - (1-\Phi)y_i = 0 \\\\
\sum_{i=1}^M [y_i-\Phi] = 0 \\\\
\Phi = \frac {1} {M} \sum_{i=1}^M y_i \\\\
$$
对$\mu_1$进行求偏导，

$$
\sum_{i=1}^M log[N(\mu_1|\Sigma)^{y_i}] = \sum_{i=1}^M y_i log[\frac {exp(- \frac 1 2 (x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1))} {(2\pi)^{p/2}(\Sigma)^{1/2}} ] = 0 \\\\
\mu_1^* = argmax_{\mu_1} \sum_{i=1}^M y_i (- \frac 1 2 (x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1)) \\\\
= - \frac 1 2 \sum_{i=1}^M (x_i^T \Sigma^{-1} - \mu_1^T \Sigma^{-1})(x_i-\mu_1) \\\\
\frac{\partial L(\theta)}{\partial \mu_1} = \sum_{i=1}^M y_i(-\Sigma^{-1}x_i + \Sigma^{-1}\mu_1^T) = 0 \\\\
\mu_1^* = \frac{\sum_{i=1}^M y_i x_i} {\sum_{i=1}^M y_i}
$$
类似的，我们可以推导出
$$
\mu_0^* = \frac{\sum_{i=1}^M (1-y_i) x_i} {\sum_{i=1}^M (1-y_i)}
$$
用$S$记录矩阵的方差，我们先对$L(\theta)$与之相关的部分进行化简，
$$
L(\theta) = \sum_{i=1}^M log[N(\mu_1|\Sigma)^{y_i}] + log[N(\mu_0|\Sigma)^{1-y_i}] + log [\Phi^{y_i}(1-\Phi)^{1-y_i}] \\\\
= \sum_{k=1} log[N(\mu_1|\Sigma)] + \sum_{k=0} log[N(\mu_0|\Sigma)] + \sum_{i=1}^M log [\Phi^{y_i}(1-\Phi)^{1-y_i}] \\\\
-------------------------------------- \\\\
\sum_{i=1}^M log N(\mu, \Sigma) = \sum_{i=1}^Mlog[\frac {exp(- \frac 1 2 (x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1))} {(2\pi)^{p/2}|\Sigma|^{1/2}} ] \\\\
= log[\frac 1 {(2\pi)^{p/2}}] + log[|\Sigma|^{- \frac 1 2}] - \frac 1 2 (x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1) \\\\
= \sum_{i=1}^M C -\frac 1 2 log|\Sigma| -\frac 1 2 (x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1) \\\\
= C - \frac 1 2 \sum_{i=1}^M log|\Sigma| -\frac 1 2 \sum_{i=1}^M (x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1) \\\\
= C - \frac M 2 log|\Sigma| -\frac 1 2 \sum_{i=1}^M tr[(x_i-\mu_1)^T \Sigma^{-1} (x_i-\mu_1)] \\\\
= C - \frac M 2 log|\Sigma| -\frac 1 2 \sum_{i=1}^M tr[(x_i-\mu_1)^T  (x_i-\mu_1) \Sigma^{-1}] \\\\
= C - \frac M 2 log|\Sigma| -\frac 1 2 tr[\sum_{i=1}^M (x_i-\mu_1)^T  (x_i-\mu_1) \Sigma^{-1}] \\\\
= C - \frac M 2 log|\Sigma| -\frac M 2 tr[S*\Sigma^{-1}] \\\\
$$

用$M_k$表示各类数据的个数，对$\Sigma$进行求偏导，
$$
L(\theta) = - \frac {M_1} 2 [log|\Sigma| - tr(S_1 * \Sigma^{-1})] - \frac {M_0} 2 [log|\Sigma| - tr(S_0 * \Sigma^{-1})] \\\\
= - \frac {M} 2 log|\Sigma| - \frac {M_1} 2  tr(S_1 * \Sigma^{-1}) - \frac {M_0} 2  tr(S_0 * \Sigma^{-1}) \\\\
-------------------------------\\\\
\frac {\partial L(\theta)}{\partial \Sigma} = - \frac {1} 2 [(M \Sigma^{-1}) - M_0 S_0^T\Sigma^{-2} - M_1 S_1^T\Sigma^{-2}] = 0 \\\\
M \Sigma = M_1 S_0 + M_1 S_1 \\\\
\Sigma^* = \frac {M_1 S_0 + M_1 S_1} {M}
$$


## 适用场景

无。

**优点**

1. 鲁棒性较好

**缺点**

1. 需要数据服从分布（具有严格假设）
2. 参数较多，计算相对比较复杂





# Reference

- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007