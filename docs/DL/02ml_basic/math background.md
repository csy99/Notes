# 记号和术语

$X \in R^{N*p} = (x_1, x_2, .., x_N)^T $

$x_i = (x_{i1}, ..., x_{ip})^T$

表示数据共有N个样本，每个样本的维度是p。

iid，即Independent and identically distributed，表示一组随机变量中每个变量的概率分布都相同，且这些随机变量互相独立。

MLE，即maximum likelihood estimation，最大似然估计，用来估计一个概率模型的参数的一种方法。

MAP，即Maximum a posteriori estimation，最大后验概率，考虑了被估计量的先验概率分布。

马氏距离，即Mahalanobis Distance，是一种距离的度量，可以看作是欧氏距离的一种修正，修正了欧式距离中各个维度尺度不一致且相关的问题。



# 统计与概率

### 频率派/统计学

$p(x|\theta)$ 中的$\theta$是一个常量。假设iid，观测到数据的N个样本来说的概率是$p(X|\theta) = \prod_{i=1}^N p(x_i|\theta))$。可以采用MLE逆向求解$\theta$。

$\theta^* = argmax_{\theta} logp(X|\theta) = argmax_{\theta} \sum_{i=1}^N log p(x_i|\theta)$

### 贝叶斯派/概率学

$p(x|\theta)$ 中的$\theta$是一个变量，服从一个先验分布$p(\theta)$。依赖数据集参数的后验概率可以写成

$p(\theta|X) = p(X|\theta)*p(\theta)/p(X) = p(X|\theta)*p(\theta)/\int_{\theta}[p(X|\theta)*p(\theta)]d\theta$

可以用MAP求解$\theta$。

$\theta^* = argmax_{\theta} p(\theta|X) = argmax_{\theta} p(X|\theta)*p(\theta)$



# 高斯分布

在iid的条件下，$x \sim N(\mu, \Sigma)$，$\Sigma = \sigma^2$

高斯分布的概率密度函数PDF可以写作$p(x|\mu, \Sigma) = 1/[(2\pi)^{p/2}|\Sigma|^{1/2}] * e^{-0.5(x-\mu)^T \Sigma^{-1}(x-\mu)}$

**缺点**

1. $\Sigma$有$\frac {p^2 + p}{2}$个参数，对于高维的$x$计算过于复杂。
2. 难以处理多个峰值的数据

## 一维

此时$p$ = 1, 
$$
log P(X|\theta) = log \prod_{i=1}^N (X|\theta) = \sum_{i=1}^N log p(x_i|\theta) \\\\
= \sum_{i=1}^N 1/[(2\pi)^{1/2}|\Sigma|^{1/2}] * exp[{-\frac 1 2 (x_i-\mu)^T \Sigma^{-1}(x_i-\mu)}] \\\\
= \sum_{i=1}^N [log \frac{1}{(2\pi)^{1/2}} + log \frac{1}{\sigma} - \frac{(x_i-\mu)^2}{2\sigma^2}]
$$
首先对$\mu$取极值
$$
\mu_{MLE} = argmax_\mu log p(X|\theta) \\\\
= argmax_\mu -\sum_{i=1}^N \frac{(x_i - \mu)^2}{2\sigma^2} \\\\
= argmin_\mu \sum_{i=1}^N (x_i - \mu)^2 \\\\
$$
进行求导，
$$
\frac{\part }{\part \mu} \sum_{i=1}^N (x_i - \mu)^2 = 0 \\\\
-2*(\sum_{i=1}^N (x_i - \mu))= 0 \\\\
\sum_{i=1}^N x_i - \sum_{i=1}^N \mu = 0 \\\\
\mu_{MLE} = \frac{\sum_{i=1}^N x_i}{N}
$$

发现求出来的$\mu_{MLE}$的期望实际上就是数据的均值，所以是无偏差的；方差是$\frac{\sigma^2}{N-1}$。

其次对$\sigma$取极值
$$
\sigma_{MLE}^2 = argmax_{\sigma} log p(X|\theta) \\\\
= argmax_{\sigma} \sum_{i=1}^N(-log\sigma - \frac{(x_i-\mu)^2}{2\sigma^2})
$$
用L标记算式整个部分，进行求导，
$$
\frac{\part L}{\part \sigma} = 0 \\\\
\sum_{i=1}^N-\frac{1}{\sigma} - (-2)*\frac{1}{2}*(x_i-\mu)^2*\sigma^{-3}= 0 \\\\
\sum_{i=1}^N-\sigma^2 + (x_i-\mu)^2 = 0 \\\\
-\sum_{i=1}^N\sigma^2 + \sum_{i=1}^N(x_i-\mu)^2 = 0 \\\\
\sum_{i=1}^N\sigma^2 = \sum_{i=1}^N(x_i-\mu)^2 \\\\
\sigma^2_{MLE} = \frac{\sum_{i=1}^N (x_i-\mu)^2}{N} \\\\
$$
求出来的$\sigma^2_{MLE}$通常被称为有偏估计，因为$E[\sigma^2_{MLE}] = \frac{N-1}{N}\sigma^2$。

## 多维

公式的前半部分跟x没有关系，我们可以直接看后半部分。将$(x-\mu)^T \Sigma^{-1}(x-\mu)$看做$x$与$\mu$之间的马氏距离。$\Sigma$是半正定的对称矩阵。当$\Sigma = I$，马氏距离退化成为欧氏距离；在这里我们只考虑$\Sigma$正定的情况。
$$
\Sigma = U \Lambda U^T \\\\
 = [u_1\;u_2\;...\;u_p]\ Diag([\lambda_1\;\lambda_2\;...\;\lambda_p])\ [u_1\;u_2\;...\;u_p]^T \\\\
 = [u_1\lambda_1\;u_2\lambda_2\;...\;u_p\lambda_p]\ [u_1\;u_2\;...\;u_p]^T \\\\
 = \sum_{i=1}^p u_i \lambda_i u_i^T
$$
容易推导出$\Sigma^{-1} = \sum_{i=1}^p u_i \frac 1 {\lambda_i} u_i^T$。令$h_i = (x-\mu)^T u_i$。
$$
(x-\mu)^T \Sigma^{-1}(x-\mu) \\\\
= (x-\mu)^T \sum_{i=1}^p u_i \frac 1 {\lambda_i} u_i^T (x-\mu) \\\\
= \sum_{i=1}^p (x-\mu)^T u_i \frac 1 {\lambda_i} u_i^T (x-\mu) \\\\
= \sum_{i=1}^p h_i \frac 1 {\lambda_i} h_i^T \\\\
= \sum_{i=1}^p \frac {h_i^2} {\lambda_i} \\\\
$$

如果将这个公式（马氏距离）取一个定值，不难发现代表一个椭圆曲线，表示一个切面，类似于等高线。



# Reference

- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007
- CS540 Intro to AI @UW-Madison, Jerry Zhu