# 术语

样本空间(sample space)：$\Omega$，包含了所有可能出现的结果的集合。比如在掷一次骰子的样本空间可以用{1,2,3,4,5,6}表示。

事件集(event space): $F$，a collection of subsets of $\Omega$，用来表示出现的结果。事件集未必是样本空间中的单一元素，也可以是复杂元素。比如在掷一次骰子的样本空间中，可以用{1,3,5}表示结果为奇数的事件。

概率函数(probability function): $P$，该函数完成了从事件到该事件发生概率的映射。



# 概率法则

### 贝叶斯

A的先验概率(prior probability of A): P(A)

A的后验概率(posterior probability of an event A given B): P(A|B)
$$
P(A|B) = \frac {P(B|A)P(A)} {P(B)}
$$

### 独立事件

事件$A_1, A_2,\ ...\ , A_n$相互独立，当且仅当该事件集合的**所有子集**满足条件$P(A_{i1}, A_{i2},\ ...\ , A_{ik}) = \prod_{j=1}^k P(A_{ij})$



# 随机变量

一般来说，我们使用大写字母表示随机变量本身，用对应的小写字母代表该变量的取值。

可以从CDF分辨一个随机变量是离散变量、连续变量、抑或是两者都不是。

<img src="https://i.postimg.cc/qqvQMqVs/RV.png" height=400>

## 离散变量

满足条件$P(X \in \mathcal X) = 1$ for some countable set $\mathcal X \sub R$。

离散变量可以被其概率质量函数充分说明。

### 概率质量函数

probability mass function (pmf)。定义$p(x) = P(X=x) \ \forall \ x \in X$。

性质：

1. $p(x) \ge 0$
2. $\sum_{x \in X} p(x) = 1$

我们常用记号$X \sim p(x)$来表示X的pmf是p(x)。

### 累积分布函数

cumulative density function (cdf)。定义$F(x) = P(X \le x)$。

**性质**

1. $F(x) \ge 0$，且单调非递减
2. $lim_{x->\infty} F(x) = 1$，$lim_{x->-\infty} F(x) = 0$

3. $F(x)$ 是右连续的，即$lim_{x->a^+} F(x) = F(a)$
4. $P(X=a) = F(a) \ - \ lim_{x->a^-} F(a)$

### 经典的离散变量

#### Bernoulli

$p(x) = px + (1-p)(1-x); \ x \in \{0,1\}$

应用场景为投篮投进的概率。

#### Geometric

$p(x) = p(1-p)^x$

应用场景为抛硬币直到看到一次正面朝上的概率。

#### Binomial

$p(x) = C(n, k)*p^k(1-p)^{n-k}$

应用场景为连续抛n次硬币看到k次正面朝上的概率。

#### Poisson

$p(x) = \frac {\lambda^x} {x!} e^{-\lambda}; \lambda > 0$

应用场景为在给定时间段内事件的数量。

#### Categorical

可以自己根据场景定义pmf。

## 连续变量

### 概率密度函数

probability density function (pdf)。定义$f(x) = \frac {dF(x)} {dx}$。

**性质**

1. $f(x) \ge 0$
2. $\int_{-\infty}^{\infty} f(x) dx = 1$，同理$P(X \le a) = \int_{-\infty}^{a} f(x) dx$
3. $P(X \in A) = \int_{x \in A} f(x) dx$

我们常用记号$X \sim f(x)$来表示$X$的pdf是$f(x)$。

### 累积分布函数

与离散变量的CDF部分相同。

### 经典的连续变量

#### 高斯Gaussian

$X \sim \mathcal N(\mu, \sigma^2)$
$$
f(x) = \frac {1} {\sqrt{2\pi \sigma^2}} * e^{-\frac {(x-\mu)^2} { 2\sigma^{2}}}
$$

#### Logistic

$X \sim logistic(\mu=0, s=0)$
$$
f(x) = \frac {e^{-x}} {(1+e^{-x})^2}
$$

#### Uniform

$X \sim U[a,b]$
$$
f(x) = \frac 1 {b-a}; \ for \ a \le \ x \le b
$$

$var(X) = \frac {(b-a)^2} {12}$

#### Exponential

$X \sim Exp(\lambda); \lambda > 0$
$$
f(x) = \lambda e^{-\lambda x}; \  x \ge 0
$$

#### Laplace

$X \sim Lap(\mu, b); \ b > 0$
$$
f(x) = \frac 1 {2b} e^{-\frac{|x - \mu|} {b}}
$$



# 期望&方差&矩

### 期望

假设$X \sim p(x)$，则$E[X] = \sum_{x \in X} xp(x)$。容易得到$E[g(X)] = \sum_{x \in X} g(x)p(x)$。

假设$X \sim f(x)$，则$E[X] = \int_{-\infty}^{\infty} xf(x)$。容易得到$E[g(X)] = \int_{-\infty}^{\infty} g(x)f(x) dx$。

需要注意的是，期望是有可能发散的。比如$g(x) = x^{-2}; \ x \ge 1$的期望就是正无穷。

**性质**

1. 线性，$E[a*g(X) + b*h(X) + c] = a*E[g(X)] + b*E[h(X)] + c$
2. 可转换性，如果$Y = g(X)$，那么$E[Y] = E[g(X)]$

### 方差

方差$var(X)$，有时候也用$D(X)$表示。

$D[X] = E[(X - E[X])^2] = E[X^2] - (E[X])^2$。数学推导见下，
$$
D[X] = \sum_{i=1}^n (x_i - \mu)^2 p_i \\\\
= \sum_{i=1}^n x_i^2 p_i - 2\mu \sum_{i=1}^n x_i p_i + \mu^2 \sum_{i=1}^n p_i \\\\
= \sum_{i=1}^n x_i^2 p_i - 2 \mu^2 + \mu^2 \sum_{i=1}^n p_i \\\\
= \sum_{i=1}^n x_i^2 p_i - \mu^2 \\\\
= E[X^2] - (E[X])^2
$$
**性质**

1. $D[ax+b] = a^2*D(x)$

### 矩

英文是moment，有时候被称为动差。

$i$阶矩被定义为$E[X^i]$，可以发现一阶矩正好就是期望。0阶矩被定义为1。

### 概率的界

#### Markov不等式

假设$X$是一个非负随机变量(RV)，那么对于任何非负的实数$a$有$P(X \ge a) \le \frac {E[X]} a$。
$$
a*\mathbb{I}_{\{X \ge a\}} \le X \\\\
E[a*\mathbb{I}_{\{X \ge a\}}] \le E[X] \\\\
a E[\mathbb{I}_{\{X \ge a\}}] \le E[X] \\\\
a \sum_{x \in X} [P(X=x) \mathbb{I}_{\{X \ge a\}}] \le E[X] \\\\
a P(X \ge a) \le E[X] \\\\
$$

#### Chebyshev不等式

假设$X$是一个随机变量(RV)，那么对于任何实数$a>1$，有$P(|X-E[X]| \ge a\sigma) \le \frac 1 {a^2}$.



# 联合概率

假设iid，$p(x, y) = P(X=x, Y=y)$，$(X,Y) \sim p(x,y)$。

### 联合概率质量函数

边缘分布(marginals)可以表示成$p(x) = \sum_{y \in \mathcal Y} p(x, y)$

$X$, $Y$相互独立<=>$p(x, y) = p(x)p(y) \ \forall \ x \in \mathcal X, y \in \mathcal Y$

### 联合累积分布函数

$F(x,y) = P(X \le x, Y \le y) \ \forall \ x \in R, y \in R$

容易得到$P(a < X \le x, b < Y \le y) = F(b,d) - F(a,d) - F(b,c) + F(a,c)$。

**性质**

1. 在x和y方向均不递减
2. $lim_{x->+\infty} F(x,y) = F(y)$

### 联合概率密度函数

$$
f(x,y) = \frac {\partial^2 F(x,y)} {\partial x \partial y}
$$

计算$X$的边缘联合概率质量函数(marginal pdf)：$f(x) = \int_{-\infty}^{\infty} f(x,y) dy$

**性质**

1. $P((X,Y) \in A) = \int_{(x,y) \in A} f(x,y) dxdy$

### 联合高斯

Jointly Gaussian，也被称为Bivariate Gaussian。定义$\rho$为关联系数(correlation coefficient)。



# 变量间的相互关系

### 协方差

covariance。用于衡量两个随机变量的联合变化程度。

$cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$

如果两个变量相互独立，那么协方差是0。但是反之并不成立！如果两个变量的协方差是0，我们只能说这两个变量不相关，但是不能得出相互独立的结论。

<img src="https://i.postimg.cc/mDw8sMqZ/cov0.png" height= 300>

上面这张图就是协方差为0但变量不相互独立的例子。

我们仔细观察可以发现，方差是协方差的一种特殊情况，是变量与自身的协方差。

$var(X+Y) = var(X) + var(Y) + 2cov(X,Y)$。

我们可以用方差的公式证明这一推论。
$$
var(X+Y) = E[(X+Y)^2] - (E[X+Y])^2 \\\\
= E[X^2] + E[Y^2] + 2E[XY] - (E[X+Y])^2 \\\\
= (E[X^2] - E[X]^2 + E[X]^2) + (E[Y^2] - E[Y]^2+ E[Y]^2) + 2E[XY] - (E[X+Y])^2 \\\\
= var(X) + E[X]^2 + var(Y) + E[Y]^2 + 2E[XY] - (E[X+Y])^2 \\\\
= var(X) + var(Y) + E[X]^2 + E[Y]^2 + 2E[XY] - (E[X]+E[Y])^2 \\\\
= var(X) + var(Y) + 2E[XY] + E[X]^2 + E[Y]^2 - (E[X]+E[Y])^2 \\\\
= var(X) + var(Y) + 2E[XY] - 2E[X][Y] \\\\
= var(X) + var(Y) + 2cov(X,Y)
$$
**性质**

1. 对称性
2. $cov(aX, bY) = ab \ cov(X,Y)$

### 相关

correlation。显示两个随机变量之间线性关系的强度和方向。如果变量之间有很强的关系但不是线性关系，correlation也很可能是0。

$E[XY] = \sum_{x \in X} \sum_{y \in Y} xyp(x,y)$

<img src="https://i.postimg.cc/3Nf1yhYP/corr1.png" height= 200>

上面图示分别对应correlation值接近0，1，-1.

### 相关系数

Correlation Coefficient。一般指的都是皮尔森系数。
$$
\rho = \frac {cov(X, Y)} {\sqrt{var(X)var(Y)}}
$$
**性质**

1. 对称性



# 协方差矩阵

一个向量由多个随机变量组成（默认是列向量），用$v$或者$x$表示。

随机向量$ v $的协方差矩阵是所有RV对之间的协方差的矩阵。实际上，我们可以将其视为对单个RV的方差的扩展。

<img src="https://i.postimg.cc/501nTpsk/cov-mat.png" height=150>

我们可以从定义出发进行推导得到一个推论，注意下面多处包含的是向量的外积：
$$
\Sigma_{v} = E[(v-\mu_v)(v-\mu_v)^T] \\\\
= E[vv^T - v\mu_v^T - \mu_vv^T + \mu_v\mu_v^T] \\\\
= E[vv^T] - E[v\mu_v^T] - E[\mu_v v^T] + E[\mu_v \mu_v^T] \\\\
= E[vv^T] - E[v]\mu_v^T - \mu_v E[v^T] + \mu_v\mu_v^T \\\\
= E[vv^T] - \mu_v \mu_v^T - \mu_v \mu_v^T + \mu_v\mu_v^T \\\\
= E[vv^T] - \mu_v \mu_v^T
$$
**性质**

1. 对称性
2. 半正定性



# 参数估计

### 最大后验概率

Maximum-a-posteriori (MAP)。对于分类问题，MAP将0-1损失最小化。

假设$x,y$都是离散的。
$$
\hat y = g(x) = argmax_y p(y|x) \\\\
= argmax_y p(x,y) \\\\
= argmax_y p(x|y)p(y) \\\\
$$
假设$x$是连续的，$y$是离散的。
$$
\hat y = g(x) = argmax_y p(y|x) \\\\
= argmax_y f(x|y)p(y)
$$
**缺点**

1. 随机变量相互独立的假设通常不成立
2. 训练集中未出现某个值的样本导致概率为0，可以通过smoothing解决

### 最大似然估计

Maximum Likelihood Estimation (MLE)。在MAP的基础上，我们假设$y$的先验概率都相同。
$$
\hat y = argmax_y p(y|x) \\\\
= argmax_y p(x|y) \\\\
$$
用$y$表示会显得完全随机，有可能该参数是服从某个分布的，所以常用$\theta$表示。在保证独立的情况下，可以得到
$$
p_\theta(x) = \prod_{i=1}^N p(x_i) \\\\
$$
假设进行$n$次伯努利实验，观察到$k$次成功。我们想要预测其成功概率，用$\theta$来表示。当$\theta=k/n$时有最大值。
$$
argmax_\theta \theta^k(1-\theta)^{n-k} \\\\
= argmax_\theta log[\theta^k(1-\theta)^{n-k}] \\\\
= argmax_\theta[k log\theta + (n-k)log(1-\theta)] \\\\
k/\theta - (n-k)/(1-\theta) = 0 \\\\
\theta^* = k/n
$$
同理，在$\theta$服从高斯分布的情况下，我们可以得到$\theta^* = \frac 1 n \sum x_i$。

### 最小二乘法

Minimum Mean Squared Error (MMSE)。最小化平方误差函数。
$$
\hat y = f(x) = E[y|x] \\\\
$$

考虑两个向量 $x$ 和$y$ 有联合分布 $p(x,y)$。我们可以得到，
$$
E[(y - E[y|x])^2] = E[y^2 - 2yE[y|x] + (E[y|x])^2] \\\\
= E[y^2] - 2E[yE[y|x]] + E[(E[y|x])^2] \\\\
= E[y^2] - 2E[E[yE[y|x]|x]] + E[(E[y|x])^2] \\\\
= E[y^2] - 2E[E[y|x]E[y|x]] + E[(E[y|x])^2] \\\\
= E[y^2] - E[(E[y|x])^2] \\\\
= E[E[y^2|x]] - (E[E[y|x]])^2 \\\\
= E[var(y|x)]
$$



# 条件独立

我们认为当前概率可能与其前面发生事件的概率相关（即条件），$p(x) = p(x_1)p(x_2|x_1)p(x_3|x_1, x_2)\ ...\ p(x_n|x_1, \ ...\ x_{n-1})$。
$$
p(x) = \prod_{j=1}^N p(x_j | x_1, x_2, \ ...\ ,x_{j-1})
$$
需要注意的是，分解形式并不唯一，我们也可以调换分解的顺序。不过，上式是最常见的形式。

我们将X和Y两个RV相互独立记作$X \bot Y$。

条件独立的英文是conditional independence。

$X_1$和$X_2$在$X_3$发生时条件独立是$p(x_1, x_2|x_3) =p(x_1|x_3)p(x_2|x_3)$的充要条件，记作$X_1 \bot X_2 | X_3$。我们也可以推出此时$p(x_1|x_2, x_3) = p(x_1|x_3)$。

我们来看一个例子。我们抛一枚质地均匀的硬币两次。结果分别记为$X_1$和$X_2$，令$X_3$为前两者的异或结果。容易得到$p(x_1) = p(x_2) = p(x_3) = \frac {1} {2}; x = \{0,1\}$。我们发现任意两个RV之间都是独立的，例如$X_1$，$X_3$。但是当$X_3$发生时（已知$x_3$），$X_1$和$X_2$之间并非独立的。

再来看一个例子。我们抛一枚质地均匀的硬币，结果记为$X_1$。如果结果为1，再抛一次硬币结果记为$X_2$，否则将$X_2$设为0。令$X_3$等于$X_2$。此时当$X_2$发生时，$X_1, X_3$是独立的（条件独立），即$X_1 \bot X_3 | X_2$。

| $X_1$ | $X_2$ | $X_3$ | $p(x_1, x_2, x_3)$ |
| ----- | ----- | ----- | ------------------ |
| 0     | 0     | 0     | $\frac{1} {2}$     |
| 1     | 0     | 0     | $\frac{1} {4}$     |
| 1     | 1     | 1     | $\frac{1} {4}$     |

### 概率图模型

普通的图由节点和边的集合构成。$G := (V, E)$。通常用边表(edge list)表示。

应用包括社区发现(community detection)，图向量嵌入(graph embeddings)等。

概率图模型是用图论方法以表现数个独立随机变量之间关联的一种建模方法。形式通常是有向无环图(directed acyclic graphical models, DAGs)。我们也可以称为贝叶斯网络(bayesian networks)。我们用$x \sim p(x)$联系到图的性质中，使用一个节点代表一个RV，使用边代表不同RV间的依赖，即$p(x) = \prod_j p(x_j | parents\ of\ x_j)$。通常情况下，我们对节点进行编号时，父节点会被先编号。

我们可以得到$X_i \bot X_j | \{parents\ of X_i\}\ \forall \ j < i \ \& \ X_j \notin \{parents\ of X_i\} $。

<img src="https://i.postimg.cc/tRd4Hkxp/probablistic-graphical-model.png" height=300>

例如，在上面图示中，$X_7 \bot X_3 | X_5, X_6$。

由此，我们可以得到结论
$$
p(x) = \prod_{j=1}^N p(x_j | parents\ of\ x_j)
$$

### 马尔科夫链

这是概率图模型的一种特殊情况，即下一状态的概率分布只由当前状态决定，在时间序列中它前面的事件均与之无关。$p(x) = p(x_1)p(x_2|x_1)p(x_3|x_2)...p(x_n|x_{n-1})$。

<img src="https://i.postimg.cc/bvvmqmJM/markov-chain.png" height=80>



# 信息熵

英文是Entropy。对于每一个事件，我们从它的发生能够获取到的信息是$log_2(\frac 1 {P(A)})$。这一个公式其实是符合我们的直觉的。如果一个不寻常事件它发生了，透露的信息应该比常见事件透露的信息更多。

信息熵的定义如下，
$$
H(X) = \sum_{i=1}^mE[log_2 \frac{1} {p(x_i)}] = -\sum_{i=1}^m p(x_i) log_2 p(x_i)
$$

**性质**

1. $H(x) \ge 0$
2. $H(x) \le log_2(|\mathcal X|)$

通过上式，我们可以推导出联合信息熵
$$
H(X, Y) = -\sum_{i=1}^m\sum_{j=1}^n p(x_i, y_j) log_2 (p(x_i, y_j))
$$
同理，可以得到条件信息熵
$$
H(X|Y) = -\sum_{y} p(y) H(X|Y=y) \\\\
$$
实际上，通过信息熵的链式法则，我们可以使用条件信息熵得到联合信息熵。

$H(X, Y) = H(X) + H(Y|X)$

推导如下，
$$
H(X, Y) = -\sum_{x}\sum_{y} p(x, y) log_2 (p(x, y)) \\\\
= -\sum_{x}\sum_{y} p(x, y) log_2 (p(x) * p(y|x)) \\\\
= -\sum_{x}\sum_{y} p(x, y) (log_2p(x) + log_2p(y|x)) \\\\
= -\sum_{x}\sum_{y} p(x, y) log_2p(x) -\sum_{x}\sum_{y} p(x, y) log_2p(y|x) \\\\
= -\sum_{x} p(x) log_2p(x) -\sum_{x}\sum_{y} p(x)p(y|x) log_2p(y|x) \\\\
= H(X) -\sum_{x}p(x)H(Y|X=x) \\\\
= H(X) + H(Y|X) \\\\
$$
通过多次使用该链式法则，我们可以得到
$$
H(X_1, X_2,\ ...\ , X_n) = \sum_{i=1}^n H(X_i | X_1, X_2,\ ...\ , X_{i-1})
$$


**性质**

1. $H(X) \ge H(X|Y)$，已知条件减小了不确定性
2. $H(X) = H(X|Y); X \bot Y$
3. $H(X, Y) = H(X) + H(Y); X \bot Y$
4. $H(X, Y) = H(X); Y = g(X)$



# 互信息

互信息被定义为$I(Y;X) = H(Y) - H(Y|X)$，用于表示由于已知$X$，$Y$的不确定性被减小了的部分。

**性质**

1. $I(Y;X) = I(X;Y)$
2. $I(Y;X) = \sum_{x, y} p(x, y) log(\frac{p(x, y)} {p(x)p(y)})$

从下面这张韦恩图，我们可以直观地看见各个部分。

<img src="https://i.postimg.cc/KckDSW7D/mutual-information.png" >



# 多变量高斯分布

我们考虑一个RV向量$x$，期望是$\mu$，有协方差矩阵$\Sigma_x$。可以得到联合高斯RV如下。

$x \sim \mathcal N(\mu, \Sigma)$
$$
f(x) = \frac {1} {\sqrt{(2\pi)^{n} |\Sigma|}} * e^{-\frac {(x-\mu)^T\Sigma^{-1}(x-\mu)} { 2}}
$$
因为$\Sigma$是半正定的，我们知道$\Sigma^{-1}$是半正定的。指数部分的等高线我们可以得到一个椭圆如下。
$$
(x-\mu)^T\Sigma^{-1}(x-\mu) \ge 0; \ if\ x- \mu \ne 0 \\\\
(x-\mu)^T\Sigma^{-1}(x-\mu) = c
$$
在最简单的情况下，$x \sim \mathcal N(0, I)$，我们得到一个圆心在原点的圆。在一般情况下，我们需要对协方差矩阵进行分解。
$$
\Sigma = U\Lambda U^T = \sum \lambda_i uu^T \\\\
(x-\mu)^T\Sigma^{-1}(x-\mu) = \sum \frac 1 {\lambda_i} (x-\mu)^T u_i u_i^T (x-\mu) \\\\
= \sum \frac {y_i^2} {\lambda_i}
$$
这里的$y$并没有特殊含义，只是为了书写简便，对表达式进行了简化。

**性质**

1. 线性变换：已知$x \sim \mathcal N(\mu, \Sigma)$，则$Ax \sim \mathcal N(A \mu, A \Sigma A^T)$

2. 如果RV之间无关联，即相互独立，则$\Sigma$是对角矩阵，$f(x) = \prod_i f(x_i)$

3. 边缘概率分布(marginal)也是高斯随机变量
4. 条件概率(conditional)也是高斯随机变量

对于性质1，在求边缘分布(marginal)的时候特别有用。我们一般需要通过对其他特征维度进行积分的方法求解，但我们也可以利用性质1构造一个巧妙的$A$来求解。比如此时有三维特征，我们想求$p(x_2)$，那么我们将$A$设置成$[0\ 1\ 0]$即可。这个方法帮助我们省去积分的繁琐步骤。

对于性质3，要注意它反过来并不成立。即，假设边缘概率分布都是高斯随机变量，并不意味着由这些随机变量组成的RV服从高斯分布。

我们再对性质4做一些说明。通过线代，我们知道可以对矩阵做划分如下。

<img src="https://i.postimg.cc/9f9dJFmS/conditional-gaussian.png">

我们将$\Sigma_{x_2|x_1}$称为$\Sigma_{11}$的舒尔补（Schur complement）。从操作上看其实非常容易理解，将$\Sigma$取逆，将和已知条件相关的行和列去掉，再将得到的小矩阵取逆。这里的描述有些抽象，具体过程请参阅 (Murphy, 2012, 4.1-4.2)。

$\Sigma_{x_2|x_1} = \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$

由舒尔补，我们可以得到
$$
x_2|x_1 \sim \mathcal N(\Sigma_{21}\Sigma_{11}^{-1}(x_1 - \mu_1) + \mu_2, \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12})
$$

同理可得
$$
x_1|x_2 \sim \mathcal N(\Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2) + \mu_1, \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})
$$



# 高斯判别分析

我们假设观测数据来自于条件高斯分布。
$$
p(x | y = 0) \sim \mathcal N(\mu_0, \Sigma_0) \\\\
p(x | y = 1) \sim \mathcal N(\mu_1, \Sigma_1) \\\\
$$
我们通过将两者的概率相除，然后取对数，就可以得到对数似然比（log likelihood ratio）。大于0的归为一类，小于0的归为另一类。

分类边界为$x^T B x + w^T x = c$。如果两个高斯分布的协方差矩阵相同，那么分类边界可以简化成为$w^T x= c$。这里我们假设两类的先验概率相同。


$$
f(x) = log(\frac{f(x|Y=0)} {f(x|Y=1)}) \\\\
= log(f(x|Y=0)) - log(f(x|Y=1)) \\\\
= log(\frac {1} {\sqrt{(2\pi)^{2} |\Sigma_0|}} * e^{-\frac {(x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)} {2}}) - log(\frac {1} {\sqrt{(2\pi)^{2} |\Sigma_1|}} * e^{-\frac {(x-\mu_1)^T \Sigma_1^{-1} (x-\mu_1)} {2}}) \\\\
= log(\frac {1} {\sqrt{(2\pi)^{2} |\Sigma_0|}}) -\frac {(x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)} {2} - log(\frac {1} {\sqrt{(2\pi)^{2} |\Sigma_1|}}) + \frac {(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)} {2} \\\\
= log(\sqrt{\frac { |\Sigma_1|}{|\Sigma_0|}}) + \frac{1} {2} [(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) - (x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)] \\\\
= \frac{1} {2} log({|\Sigma_1|}) - \frac{1} {2} log({|\Sigma_0|}) + \frac{1} {2} [(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) - (x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)] \\\\
$$

边界可以通过乘以常数得到
$$
log({|\Sigma_1|}) - log({|\Sigma_0|}) + (x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) - (x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0) = 0 \\\\
x^T (\Sigma_1^{-1} - \Sigma_0^{-1}) x + 2(\mu_0^T\Sigma_0^{-1} - \mu_1^T \Sigma_1^{-1})x = log({|\Sigma_0|}) - log({|\Sigma_1|}) + \mu_0^T\Sigma_0^{-1}\mu_0 - \mu_1^T\Sigma_1^{-1}\mu_1 \\\\
$$

其中的参数如下
$$
B = \Sigma_1^{-1} - \Sigma_0^{-1} \\\\
w = 2(\Sigma_0^{-1}\mu_0 -  \Sigma_1^{-1}\mu_1) \\\\
c = log({|\Sigma_0|}) - log({|\Sigma_1|}) + \mu_0^T\Sigma_0^{-1}\mu_0 - \mu_1^T\Sigma_1^{-1}\mu_1 \\\\
$$

### 参数估计

下面是对数据中某个分类而言的，即$x_i$都属于同一类别。对一组观测到的数据$x_1, \ ...\ , x_n$，我们假设iid。我们看到实际上算术上的平均值就是对参数的MLE估计。
$$
\hat \mu = \frac 1 n \sum x_i \\\\
\hat \Sigma = \frac 1 n \sum (x_i - \hat \mu)(x_i - \hat \mu)^T
$$

下面是证明过程。这里的$L$并不是损失Loss的意思，而是似然Likelihood。

$$
L(\mu, \Sigma) = \prod_i p_{_\mu}(x_i) = \prod_{i=1}^{n} \frac {1} {(2\pi)^{d/2} |\Sigma|^{1/2}} * e^{-\frac {(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)} {2}} \\\\
log(L(\mu, \Sigma)) = \frac {dn} {2} log(2\pi) - \frac {n} {2} log(|\Sigma|) - \frac {1} {2} \sum_{i=1}^{n}(x_i-\mu)^T\Sigma^{-1}(x_i-\mu)
$$
这里给出求导定理。
$$
\nabla_x \ a^T x = a  \\\\
\nabla_x \ x^T A x = (A + A^T)x \\\\
\nabla_X \ tr(AX) = A \\\\
\nabla_X log |X| = (X^{-1})^T
$$
我们想要求出$\mu^*$，需要对似然进行求导。我们发现log表达式第一项是常数，第二项跟$\mu$没有关系，所以直接对第三项求导。
$$
\nabla_\mu log(L(\mu)) = 0 \\\\
-2 \Sigma^{-1}x_i^T + 2 \Sigma^{-1} \mu = 0 \\\\
\Sigma^{-1} \sum_i(x_i - \mu) = 0 \\\\
\mu^* = \frac 1 n \sum_i x_i \\\\
$$
同理，我们可以求出$\Sigma^*$。这个稍微麻烦一些，需要对log表达式后两项求导。我们令$\Phi = \Sigma^{-1}$，对其进行求导。
$$
\nabla log|\Phi^{-1}| = -(\Phi^{-1})^T
$$
再解得$\Sigma$。
$$
\nabla_\Phi log(L(\Phi)) = 0 \\\\
\frac{n} {2}\Phi^{-1} - \frac{1} {2} \sum (x_i - \mu)(x_i - \mu)^T = 0 \\\\
\Sigma^* = \frac 1 n \sum_i (x_i - \mu)(x_i - \mu)^T \\\\
$$
我们无法保证协方差矩阵是满秩的（如果不是将无法取逆），所以需要加入正则化参数。$(\hat \Sigma + \lambda I)$可以在$\lambda$为正的情况下保证满秩。当然，也有很多其他的方法。

最后，我们给出log-likelihood。
$$
log L(\mu, \Sigma) = -\frac n 2 log|\Sigma| - \sum_i \frac 1 2 (x_i-\mu)^T\Sigma^{-1}(x_i-\mu)
$$


### 补充说明

上面的推导是假设了两个类别的先验概率相同。这里的推导放宽了这个限制。
$$
L(\mu, \Sigma) = \frac {p(y=0) f(x|y=0)} {p(y=1) f(x|y=1)} \\\\
= [\frac {p_0} {\sqrt{(2\pi)^{n} |\Sigma_0|}} * e^{-\frac {(x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)} { 2}}] / [\frac {p_1} {\sqrt{(2\pi)^{n} |\Sigma_1|}} * e^{-\frac {(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)} { 2}}] \\\\
= [\frac {p_0} {\sqrt{|\Sigma_0|}} * e^{-\frac {(x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)} { 2}}] / [\frac {p_1} {\sqrt{ |\Sigma_1|}} * e^{-\frac {(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)} { 2}}] \\\\

log(L(\mu, \Sigma)) = \\\\
log(\frac{p_0} {\sqrt{|\Sigma_0|}}) - log(\frac{p_1} {\sqrt{|\Sigma_1|}}) + \frac{1} {2} [(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) - (x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)] \\\\
= \frac 1 2 log(\frac{p_0^2} {|\Sigma_0|}) - \frac 1 2 log(\frac{p_1^2} {|\Sigma_1|}) + \frac{1} {2} [(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) - (x-\mu_0)^T\Sigma_0^{-1}(x-\mu_0)] \\\\
$$

经过推导，我们发现不平衡的类别先验概率只是影响了常数项。
$$
B = \Sigma_1^{-1} - \Sigma_0^{-1} \\\\
w = 2(\Sigma_0^{-1}\mu_0 -  \Sigma_1^{-1}\mu_1) \\\\
c = 2log(p(y=1)) - 2log(p(y=0)) + log({|\Sigma_0|}) - log({|\Sigma_1|}) + \mu_0^T\Sigma_0^{-1}\mu_0 - \mu_1^T\Sigma_1^{-1}\mu_1 \\\\
$$
我们来看一个具体的例子。

<img src="https://i.postimg.cc/hGpxLSyh/GDA.png" height=220>

我们来计算该分类器出错的情况。

$P(\hat y \ne y) = p(y=0)P(\hat y = 1|y=0) + p(y=1)P(\hat y = 0|y=1)$

我们可以通过画ROC曲线来评判该分类器的效果。横坐标是假正type I error (false positive)，即$P(\hat y \ne y | y = 0)$；纵坐标是假负type II error (false negative)，即$P(\hat y \ne y | y = 1)$。



# Logistic Regression

判别式模型。此模型不需要学习x的分布。
$$
\theta_i = \frac 1 {1 + e^{-x_i^Tw}}
$$
$\theta_i$是$y_i$为1的概率。当$x^T w = 0$时概率为0.5，正好是决策边界。

为了计算的方便，我们令分类的标记为-1和1(而不是0和1)。我们仍然假设每一个数据点是相互独立的。运用MLE可以得到，
$$
p_w(y_1, ..., y_n) = \prod_i \theta^{I\{y_i=1\}} (1-\theta)^{I\{y_i = -1\}}
$$
对于目标函数，
$$
w^* = argmax_w -\sum_i log(1 + e^{-y_ix_i^Tw}) \\\\
= argmin_w \sum_i log(1 + e^{-y_ix_i^Tw})
$$
可以看到，如果$x^Tw$和$y$的符号相同，说明预测准确，那么损失将非常接近于0。

目标函数是一个凸函数，所以局部最优就是全局最优。

我们可以通过证明Hessian是半正定，从而证明该函数是凸函数。亦可以通过另一些凸函数的性质证明。
$$
\frac {d^2} {dz^2} log(1 + e^{-z}) = \frac {e^z} {(e^z+1)^2}
$$
上面证明了一维成立。同时，我们知道线性变换是保留凸函数性的，并且非负的凸函数之和也为凸函数。得证，该函数是凸函数。

最后，给出一个SGD的结论。
$$
\nabla_w f(w) = \sum_i \frac {-y_i} {1 + e^{y_i w^T x_i}} x_i
$$



# 潜变量模型

目标是通过无标签的训练数据学习$p(x)$。既然是非监督学习，我们对$p(x)$更感兴趣，而不是$p(x,y)$。假设$z$是潜在变量，$p(x) = \int p(x,z) dz$。例如，在MNIST中$z$就是数字（不过在此处是无标签的）。在$z$是离散的情况下，可以由混合模型来解释。

<img src="https://i.postimg.cc/hjpHgPSw/latent-variable.png" height=200>

### 应用场景

聚类、密度预测、社区发现等。



# 高斯混合模型

即Gaussian Mixture Models (GMM)。这是潜变量模型的一种，是最常见的形式之一。$z$代表该数据点是由某一个高斯分布产生的。

$p(x) = \sum_{k=1}^K \pi_k \mathcal N(x| \mu_k, \Sigma_k)$

$\pi$在这里是指该点属于哪一个高斯分布的先验概率。除次之外，我们还需要找到每一个高斯分布的参数，即均值和协方差矩阵。下面介绍KMeans和EM两种方法。

## K-平均演算法

我们可以通过K-Means (KMeans)得到每个类的聚类中心$\mu_k$。

### 过程

1. 初始化聚类中心$\mu_k$
2. 将每个数据点分配到离自己最近的聚类中心$z_i = arg min_k ||x_i - \mu_k||_2^2$
3. 更新聚类中心$\mu_k = \frac 1 {N_k} \sum_{i:z_i=k} x_i$

<img src="https://i.postimg.cc/1tW06vs4/kmeans-1.png" height=200>

由于KMeans更擅长发现凸类型簇，聚类结果有可能出现偏差。

## 最大期望算法

Expectation Maximization (EM)。在概率模型中寻找参数最大似然估计或者最大后验估计的算法，其中概率模型依赖于无法观测的隐变量。

### 过程

初始化先验概率均等，协方差矩阵为单位矩阵(Identity Matrix)。

E-step: 

如果是软分类(soft assignment)，计算每个点到每个聚类的概率(responsibility，注意不是probability)，
$$
r_{i,k} = \frac{\pi_k \mathcal N(x| \mu_k, \Sigma_k)} {\sum_{k'} \pi_k' \mathcal N(x| \mu_k', \Sigma_k')}
$$
如果是硬分类(hard assignment)，直接分配到最有可能的中心，
$$
z_i = argmax_k \frac{\pi_k \mathcal N(x| \mu_k, \Sigma_k)} {\sum_{k'} \pi_k' \mathcal N(x| \mu_k', \Sigma_k')}
$$
M-step: 

重新估计先验、聚类中心和协方差矩阵，
$$
\mu_k = \frac 1 {\sum_i r_{i,k}} \sum_i r_{i,k}x_i \\\\
\Sigma_k = \frac 1 {\sum_i r_{i,k}} \sum_i r_{i,k} (x_i-\mu_k)(x_i-\mu_k)^T\\\\
\pi_k = \frac 1 N \sum_i r_{i,k}
$$

<img src="https://i.postimg.cc/GpPx3k2g/EM-1.png" height=400>

### 目标函数

最大化对数似然。
$$
log L(\theta) = \sum_{i=1}^n log p_\theta(x_i) = \sum_{i=1}^n log \sum_{z_i}p_\theta(x_i|z_i)
$$
 该函数有可能不是凸函数。

E-step: 对于每个数据点，在已知$\theta_{t-1}$计算$z_i$的期望值

M-step: 对于$z_i$求出$\theta_t$的最大似然












# Reference

- Probability and Information Theory in Machine Learning (ECE 601), Matthew Malloy, Fall 2020
- Machine Learning, A Probabilistic Perspective, Kevin P. Murphy, 2012



