# 前言

构造连续变量的衍生变量。在机器学习问题中，我们希望数据是服从正态分布的（或者一些常见的简单的分布）。然而，现实数据常常不服从正态分布。我们尝试进行转换，使之服从（至少更接近）正态分布。值得注意的是对于$X$和$y$的转换是不一样的。



# 基于ECDF的转换

ECDF的英文是Empirical cumulative distribution function。CDF就是统计中常说的累计分布，即$P(X \le x) = F(x)$。我们假设这个函数是可逆的。同时，我们知道如果有变量$u \sim U[0,1]$，那么
$$
P(F^{-1}(u) \le x) = F(x) \\\\
P(U \le F(x)) = F(x)
$$
我们希望对于任意分布$X \sim F(X)$，我们都可以将其转换成为另一个分布$G(X)$。所以我们需要从$X$中得到一个均匀分布。下面数学公式的构造了一个均匀分布。
$$
P(G^{-1}(F(X)) \le x) = G(x) \\\\
$$
上面概率公式中间的数学推导如下。
$$
P(F(X) \le x) = P(X \le F^{-1}(x)) \\\\
= F(F^{-1}(x)) \\\\
= x
$$
但是存在一个问题，如果$F(X)$不是单调递增的，比如在某些$x$值上函数值保持不变，那么严格来说不存在逆函数。我们需要替换成广义的逆，即最小$X$使得$y = F(X)$成立的值。

另外一个问题是，我们如何知道$X$原来的分布呢？我们只能使用分位数进行估计。



# 幂变换

在机器学习问题中，我们也会遇到$y$值不服从正态分布的情况，最常见的就是长尾数据。在统计中，幂变换是一族函数，可用于使用幂函数创建数据的单调变换。这是一种用于稳定方差，使数据更像正态分布的数据转换技术。可以提高关联度量（如变量之间的Pearson相关性）的有效性以及用于其他数据稳定程序。

## Box-Cox变换

英文是Box-Cox Transformation。要求$y > 0$。这里只介绍单变量Box-Cox变换。

当$\lambda \ne 0$时有$y = (y^{\lambda}-1)/\lambda$；当$\lambda = 0$时$y = ln(y)$。

## Yeo-Johnson变换

英文是Yeo-Johnson Transformation。
$$
y = (y^{\lambda}-1)/\lambda; \quad if \ \lambda \ne 0, y > 0 \\\\
y = ln(y+1); \quad if \ \lambda \ne 0, y > 0 \\\\
y = -[(-y+1)^{-2\lambda}-1]/(2-\lambda); \quad if \ \lambda \ne 2, y < 0 \\\\
y = -ln(-y+1); \quad if \ \lambda \ne 2, y < 0 \\\\
$$


# Reference

- NLP实战高手课，第三章，王然，极客时间
- Power transform，wikipedia