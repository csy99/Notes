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

$var(X) = p(1-p)$

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



# 联合概率

假设iid，$p(x, y) = P(X=x, Y=y)$，$(X,Y) \sim p(x,y)$。

### 联合概率质量函数

边缘分布(marginals)可以表示成$p(x) = \sum_{y \in \mathcal Y} p(x, y)$

$X$, $Y$相互独立<=>$p(x, y) = p(x)p(y) \ \forall \ x \in \mathcal X, y \in \mathcal Y$

### 联合累积分布函数

$F(x,y) = P(X \le x, Y \le y) \ \forall \ x \in R, y \in R$

容易得到$P(a < X \le b, c < Y \le d) = F(b,d) - F(a,d) - F(b,c) + F(a,c)$。

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

### 微分熵

是从以离散随机变量所计算出的夏农熵推广，以连续型随机变量计算所得之熵。
$$
H(X) = \int_x p(x) log_2(\frac 1 {p(x)}) dx
$$
假设$X \sim \mathcal N(\mu, \sigma^2)$。那么$H(X) = \frac 1 2 log_2(2\pi e \sigma^2)$比特(bits)。



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
$$

通过求对数进行简化。
$$
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



# 潜变量模型

目标是通过无标签的训练数据学习$p(x)$。既然是非监督学习，我们对$p(x)$更感兴趣，而不是$p(x,y)$。假设$z$是潜在变量，$p(x) = \int p(x,z) dz$。例如，在MNIST中$z$就是数字（不过在此处是无标签的）。在$z$是离散的情况下，可以由混合模型来解释。

<img src="https://i.postimg.cc/hjpHgPSw/latent-variable.png" height=200>

### 应用场景

聚类、密度预测、社区发现等。



# 高斯混合模型

混合模型是潜变量模型的一种，是最常见的形式之一。而高斯混合模型(Gaussian Mixture Models, GMM)是混合模型中最常见的一种。$z$代表该数据点是由某一个高斯分布产生的。$\pi$在这里是指该点属于哪一个高斯分布的先验概率。除次之外，我们还需要找到每一个高斯分布的参数，即均值和协方差矩阵。

$$
p(x) = \sum_{k=1}^K \pi_k p_k(x) \qquad \qquad (1)\\\\
p(x) = \sum_{k=1}^K \pi_k \mathcal N(x| \mu_k, \Sigma_k) \qquad (2)
$$
我们对混合模型的一般形式即(1)进行拓展，已知每一种分布单独的期望和协方差矩阵，求出$x$的期望和协方差矩阵。
$$
E[x] = \sum_x x p(x) \\\\
= \sum_x x \sum_k \pi_k p_k(x) \\\\
= \sum_k \pi_k \sum_x x p_k(x)\\\\
= \sum_k \pi_k E[p_k(x)] \\\\
= \sum_k \pi_k \mu_k
$$

$$
\Sigma_x = E[x^2] - (E[x])^2 \\\\
= E[xx^T] - (\sum_k \pi_k \mu_k)^2 \\\\
= \int xx^T p(x) dx - (\sum_k \pi_k \mu_k)^2 \\\\
= \int xx^T \sum_k \pi_k p_k(x|k) dx - (\sum_k \pi_k \mu_k)^2 \\\\
= \sum_k \pi_k \int xx^T p_k(x|k) dx - (\sum_k \pi_k \mu_k)^2 \\\\
= \sum_k \pi_k E[xx^T|k] - (\sum_k \pi_k \mu_k)^2 \\\\
= \sum_k \pi_k (E[xx^T|k] - \mu_k\mu_k^T + \mu_k\mu_k^T) - (\sum_k \pi_k \mu_k)^2 \\\\
= \sum_k \pi_k (\Sigma_k + \mu_k\mu_k^T) - (\sum_k \pi_k \mu_k)^2 \\\\
$$

下面介绍KMeans和EM两种方法。

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

在经过推导之后，可以发现当类别先验相同且方差为单位矩阵的时候，EM算法和KMeans实际上是一样的。

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

<img src="https://i.postimg.cc/GpPx3k2g/EM-1.png" height=300>

### 目标函数

最大化对数似然。
$$
log L(\theta) = \sum_{i=1}^n log p_\theta(x_i) = \sum_{i=1}^n log \sum_{z_i}p_\theta(x_i|z_i)
$$
 该函数有可能不是凸函数。

E-step: 对于每个数据点，在已知$\theta_{t-1}$计算$z_i$的期望值

M-step: 对于$z_i$求出$\theta_t$的最大似然



# 概率边界

实际情况下，我们有可能不知道$X_i$的真实分布。有一些定理可以帮助我们大致确定我们的估计有多少偏差。

### 马尔科夫不等式

英文是Markov inequality。假设$X$是一个非负随机变量(RV)，那么对于任何非负的实数$a$有$P(X \ge a) \le \frac {E[X]} a$。（$X \ge 0$，$a \ge 0$）
$$
a*\mathbb{I}_{\{X \ge a\}} \le X \\\\
E[a*\mathbb{I}_{\{X \ge a\}}] \le E[X] \\\\
a E[\mathbb{I}_{\{X \ge a\}}] \le E[X] \\\\
a \sum_{x \in X} [P(X=x) \mathbb{I}_{\{X \ge a\}}] \le E[X] \\\\
a P(X \ge a) \le E[X] \\\\
$$

### 切比雪夫不等式

英文是Chebyshev inequality。假设$X$是一个随机变量(RV)，那么对于任何实数$a>1$（否则不是很有意义，因为RHS会大于1），有$P(|X-E[X]| \ge a\sigma) \le \frac 1 {a^2}$。

这里Chebyshev概率边界只不过是Markov概率边界的拓展，假设$\hat \mu = \frac 1 n \sum_i X_i$，并且$\mu = E[X_i]$
$$
P(|\hat \mu - \mu| \ge \delta) \le \frac {\sigma^2} {n\delta^2}
$$
证明如下，（第二行是通过马尔科夫不等式推导得来）
$$
P(|\hat \mu - \mu| \ge \delta) = P((\hat \mu - \mu)^2 \ge (\delta)^2) \\\\
\le \frac {E[(\hat \mu - \mu)^2]} {\delta^2} \\\\
= \frac {var(\hat \mu)} {\delta^2} \\\\
= \frac {\sigma^2} {n\delta^2}
$$

### 样本复杂度

英文是sample complexity。为了达到错误率小于等于$\epsilon$的概率大于或等于$1 - \delta$所需要的样本数量。

定理1（上界）：我们找到了一个算法能够在小于$n(\delta, \epsilon)$个样本数量的前提下达到错误率小于等于$\epsilon$的概率大于或等于$1 - \delta$。

定理2（下界）：我们目前已经测试的所有算法达到错误率小于等于$\epsilon$的概率大于或等于$1 - \delta$都需要大于$n(\delta, \epsilon)$个样本数量。

真正需要的样本数量就是介于下界(lower limit)和上界(upper limit)之间。

拿多变量高斯分布举例。在iid的前提条件下，我们假设$X_i \sim \mathcal N(\mu, \sigma^2)$。我们对均值进行预测$\hat \mu = \frac 1 n \sum_i X_i$。那么得到的预测服从$\hat \mu \sim \mathcal N(\mu, \sigma^2/n)$。下面的推导假设$\mu = 0$。

那么如何衡量这个预测？中间的第二步利用$y = \frac{x} {\sqrt{\sigma^2/n}}$进行了换元操作。
$$
P(|\hat \mu - \mu| \ge t) = 2 \int_t^{\infty} \frac {1} {\sqrt{2\pi\sigma^2/n}} exp(-\frac{x^2} {2\sigma^2/n}) dx\\\\
= 2 \int_{t\sqrt{n/\sigma^2}}^{\infty} \frac {1} {\sqrt{2\pi}} exp(-\frac{y^2} {2}) dy\\\\
= 2Q(t\sqrt{n/\sigma^2}) \le 2*exp(-t^2n/2\sigma^2)
$$
这里的Q指的是Q函数，用于表示标准正态分布的尾部分布函数。有人也将此称为高斯尾(Gaussian tail)。
$$
Q(x) := \int_x^{\infty} \frac {1} {\sqrt{2\pi}} exp(-\frac{y^2} {2}) dy\\\\
\le \int_x^{\infty} \frac {y} {x\sqrt{2\pi}} exp(-\frac{y^2} {2}) dy \\\\
= \frac{exp(-x^2/2)} {x\sqrt{2\pi}} \\\\
\le exp(-x^2/2) \ for \ x \ge \frac{1} {\sqrt{2\pi}}
$$

### 高斯尾边界

英文是Gaussian tail bounds，就是上文提到的Q函数。令$X \sim \mathcal N(0,1)$。在这里，对$t$（也就是上文的$x$）拓宽了限制条件，具体可以使用Chernoff Bound（下文有介绍）。对于所有$t \ge 0$，
$$
P(X \ge t) \le e^{-t^2/2}
$$

### 切尔诺夫界

英文是Chernoff Bound。对于随机变量$X$和实数$s \ge 0$，有
$$
P(X \ge t) \le e^{-st} E[e^{sX}]
$$
证明如下
$$
\mathcal I\{X \ge t\} \le e^{s(X-t)} \\\\
P(X \ge t) \le E[e^{s(X-t)}] = e^{-st}E[e^{sX}]\\\\
$$

下图非常直观的显示LHS和RHS的大小关系。绿色的线代表证明中的LHS，红色的代表证明中的RHS。

<img src="https://i.postimg.cc/hGP61NFz/proof-chernoff-bound.png" height=150>

我们来看看为什么上面高斯尾函数能够拓宽限制条件。求期望需要用到矩生成函数(Moment-generating function)。
$$
P(X \ge t) \le  e^{-st}E[e^{sX}] \\\\
= e^{-st}e^{s^2/2} \\\\
\le e^{-t^2/2}
$$
最后一行我们令$s =t$。

### 霍夫丁不等式

英文是Hoeffding's Inequality。假设$X_i$都是iid的随机变量且$X_i \in [a,b]$。$\hat \mu = \frac 1 n \sum_i X_i$，并且$\mu = E[X_i]$。可以通过Chernoff Bound进行推导(选择一个合适的$s$)。
$$
P(|\hat \mu - \mu| \ge t) \le 2*exp(-\frac{2t^2n} {(b-a)^2})
$$
假设iid且$X_i \sim \mathcal N(\mu, \sigma^2)$，那么
$$
P(|\hat \mu - \mu| \ge t) \le 2*exp(-\frac{t^2n} {\sigma^2})
$$

### 大数定律

英文是Law of Large Numbers (LLN)，也被称作大数法则。根据这个定律知道，样本数量越多，则其算术平均值就有越高的概率接近期望值。当然，期望本身不能是发散的。

我们先介绍弱形式。即假设$X_i$都是iid的随机变量，那么从概率的角度$\frac 1 n \sum_i X_i$会逐渐收敛到$E[X_i]$。

证明如下，
$$
P(|\hat \mu - \mu| \ge \delta) \le \frac {\sigma^2} {n\delta^2} \\\\
lim_{n -> \infty} P(|\frac 1 n \sum_i X_i - E[X_i]| \ge \delta) \le lim_{n -> \infty} \frac {\sigma^2} {n\delta^2} = 0 \\\\
$$
我们再介绍强形式。即假设$X_i$都是iid的随机变量，那么$\frac 1 n \sum_i X_i$一定会收敛到$E[X_i]$。
$$
P(lim_{n -> \infty} \frac 1 n \sum_i X_i = E[X_i]) = 1
$$

### 中心极限定理

无人不知的Central Limit Theorem (CLT)。

假设$X_i$都是iid的随机变量，均值为$\mu$，方差为$\sigma^2$。那么$\sqrt{n}(\hat \mu - \mu)$会趋近于服从正态分布$\mathcal N(0, \sigma^2)$。

下面的图例是对抛1、2、10、50次抛硬币结果的统计。

<img src="https://i.postimg.cc/mkPM646F/CLT.png" height=160>



# K-L散度

我们使用Kullback-Leibler Divergence来衡量两个分布$p$和$q$究竟有多相似。对数函数的底不是很重要，只要在计算时保持一致即可。如果是以$e$为底，那么单位是nats；如果是以2为底，那么单位是bits。需要注意的是，需要对所有$x$满足$q(x) \ne 0$，否则K-L散度不存在。另外，K-L散度不是距离，并不具有对称性(symmetric)。
$$
D(p||q) = \sum p(x) log \frac {p(x)} {q(x)} \\\\
D(p||q) = \int p(x) log \frac {p(x)} {q(x)} dx
$$
假设有$P$和$Q$两个分布，均为iid。我们发现，当$p$是正确标签的时候，K-L散度实际上是对数似然比例的期望。
$$
D(p||q) = E_p[log \frac {p(x)} {q(x)}]
$$
我们可以通过计算$D(p||q)$找到，在服从先验假设和观测的前提下，最有可能的分布。更加便捷的是，K-L散度在p和q维度上都是凸优化问题。
$$
min_p D(p||q)
$$
**性质**

1. 非负性，当且仅当对于所有$x$有$p(x) = q(x)$的时候，散度为0

对于性质1的非负性，我们有如下证明。
$$
-D(p||q) = -\sum p(x) log \frac {p(x)} {q(x)} \\\\
= \sum p(x) log \frac {q(x)} {p(x)} \\\\
\le log (\sum p(x)\frac {q(x)} {p(x)}) \\\\
= 0
$$
证明后半句，需要用到下面提到的詹森不等式。

### 假设检验中的误差指数

英文是Error exponents in hypothesis testing，有时候也被称为Chernoff-Stein Lemma。假设观察到n个i.i.d.来自p或q的样本，并且$P(\hat y=q | y=p) \le \epsilon$，那么
$$
P(\hat y=p | y=q) \ge exp(-nD(p||q))
$$
相当于我们在固定type-I error的时候，type-II error的下降速率与指数为n和K-L散度的数相关。

误差指数的计算如下
$$
lim_{n->\infty} \frac {-ln P_{error}} {n} = D(p||q)
$$

### 詹森不等式

Jensen's inequality。对于任意凸函数$f$和随机变量$X$
$$
E[f(X)] \ge f(E[X]) \\\\
\sum p(x)f(x) \ge f(\sum xp(x))
$$
**证明**

对于任何一组$x$，有
$$
p(x_1)f(x_1) + p(x_2)f(x_2) \ge f(x_1p(x_1) + x_2p(x_2))
$$
接下来，可以使用induction证明。

### 与信息熵和互信息的关系

我们知道互信息是
$$
I(Y;X) = \sum_{x, y} p(x, y) log(\frac{p(x, y)} {p(x)p(y)})\\\\
= E[log(\frac{p(x, y)} {p(x)p(y)})]
$$
那么，我们可以将K-L散度中的$P$看成是$X$和$Y$的联合概率，$Q$看成是$X$和$Y$两个边缘分布的乘积。
$$
I(Y;X) = D(p(x,y) || p(x)p(y))
$$
由熵的性质，我们也可以推出K-L散度的非负性。我们接下来看看熵的第二个性质$H(x) \le log_2(|\mathcal X|)$。

我们假设$X$是离散且有限的，$q(x) = \frac 1 {|\mathcal X|}$，即类别概率相同。
$$
D(p||q) = \sum p(x) log \frac {p(x)} {q(x)} \\\\
= \sum p(x) (log(|\mathcal X|) + log(p(x))) \\\\
= log(|\mathcal X|) - H(X) \\\\
\ge 0
$$
我们通过上面的推导也同样得出了熵的第二个性质。最后一行可以取等号当前仅当$p(x) = \frac 1 {|\mathcal X|}$。

### 法诺不等式

英文是Fano's Inequality，有时候也称作the Fano converse。

<img src="https://i.postimg.cc/rwW8xFS8/fano-inequality.png" height=100>

假设我们有如上图所示的模型。
$$
P(\hat Y \ne Y) \ge \frac {H(Y|X) - 1} {log(|\mathcal Y|)}
$$
我们将错误定义为$E = \mathcal I_{\{\hat Y \ne Y\}}$。证明如下。
$$
H(E,Y|\hat Y) = H(Y|\hat Y) + H(E|Y,\hat Y) \\\\
= H(Y|\hat Y) \\\\
\ge H(Y|X) \\\\
$$
我们将信息熵以另一种方式拆解。
$$
H(E,Y|\hat Y) = H(E|\hat Y) + H(Y|E,\hat Y) \\\\
\le H(E) + H(Y|E,\hat Y) \\\\ \\\\
\le 1 + H(Y|E,\hat Y) \\\\
= 1 + P(E=1)H(Y|E=1,\hat Y) + P(E=0)H(Y|E=0,\hat Y) \\\\
= 1 + P(\hat Y \ne Y)H(Y|E=1,\hat Y) \\\\
\le 1 + P(\hat Y \ne Y)H(Y) \\\\
\le 1 + P(\hat Y \ne Y)log(|\mathcal Y|)
$$

### 应用

两个多变量正态分布的K-L散度求解如下。
$$
D(p(x) || q(x)) \\\\
= \int p(x)[\frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2) - \frac{1} {2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)] dx \\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}E_p[(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)] - \frac{1} {2}E_p[(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)]\\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}E_p[tr((x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2))] - \frac{1} {2}E_p[tr((x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1))]\\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_2^{-1}E_p[(x-\mu_2)(x-\mu_2)^T]) - \frac{1} {2}tr(\Sigma_1^{-1}E_p[(x-\mu_1)(x-\mu_1)^T]) \\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_2^{-1}E_p[(xx^T-2x\mu_2^T+\mu_2^T\mu_2)]) - \frac{1} {2}tr(\Sigma_1^{-1}\Sigma_1) \\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_2^{-1}(E_p[xx^T]-E_p[2x\mu_2^T]+E_p[\mu_2^T\mu_2]))-\frac{n} {2} \\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_2^{-1}(\Sigma_1+\mu_1^T\mu_1-2\mu_1\mu_2^T+\mu_2^T\mu_2)) - \frac{n} {2} \\\\
= \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_2^{-1}\Sigma_1) + (\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2) - \frac{n} {2} \\\\
$$
过程有点长，每一个步骤的依据如下

1. 将数字代入，对数和指数相抵消
2. 概率之和为1，期望的定义
3. 一个标量的迹就是它本身
4. 矩阵迹数的循环排列定律
5. 协方差矩阵的定义
6. 拆括号，$tr(I) = n $
7. 协方差矩阵的定义
8. 矩阵乘积的迹数



# 逻辑回归

逻辑回归(Logistic Regression)是判别式模型，不需要学习$x$的分布。

### 与熵的关系

#### 信息熵

计算对iid随机变量进行编码所需要的最少bits。
$$
H(p) = \sum_i p_i log(\frac{1} {p_i})
$$

#### 交叉熵

计算使用对$q$最优化的编码方式将iid随机变量$p$进行编码所需要的最少bits。

$$
H(p,q) = \sum_i p_i log(\frac{1} {q_i})
$$

如果对于某些$i$有$q_i \ll p_i$，那么$H(p,q)$会很大。

#### K-L散度

$$
D(p||q) = \sum_i p_i log(\frac{p_i} {q_i})
$$

除了将损失函数定义为交叉熵，我们亦可以定义其为K-L散度$l(y_i, \hat y_i) = D(y_i || \hat y_i)$。

#### 三者关系

$$
H(p,q) = H(p) +D(p||q)
$$

### 二分类逻辑回归

$$
\theta_i = \frac 1 {1 + e^{-x_i^Tw}}
$$

$\theta_i$是$y_i$为1的概率。$x^T w$代表到决策边界的距离（带有符号的距离）。当这个距离为0时，正好是决策边界，标记为任何一类的概率都是0.5。上面的$\sigma$公式就是将该点与决策边界的距离转化成为分类概率。

为了计算的方便，我们令分类的标记为-1和1(而不是0和1)。当$y_i$标记为-1时，我们令$p_i=[1 \quad 0]^T$；当$y_i$标记为1时，我们令$p_i=[0 \quad 1]^T$；$q_i = [1-\theta_i \quad \theta_i]^T$。

我们仍然假设每一个数据点是相互独立的。运用MLE可以得到，
$$
p_w(y_1, ..., y_n) = \prod_i \theta^{I\{y_i=1\}} (1-\theta)^{I\{y_i = -1\}}
$$
取对数有
$$
log L(w) = \sum_i I\{y_i=1\}\ log(\theta) + I\{y_i = -1\}\ log(1-\theta) \\\\
= -\sum_i H(p_i, q_i)
$$
我们将上式转化成为负对数损失(negative log loss, nll)。而且，由于$p_i$是第$i$条数据的标签（独热编码），它的信息熵是0。
$$
nll(w) = \sum_i H(p_i, q_i) \\\\
= \sum_i H(p_i) + D(p_i || q_i) \\\\
= \sum_i D(p_i || q_i)
$$
实质上，我们的目标就是最小化逻辑损失函数(logistic loss)。
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

逻辑回归没有解析解(closed-form solution)。这里给出一个SGD的结论。
$$
\nabla_w f(w) = \sum_i \frac {-y_i} {1 + e^{y_i w^T x_i}} x_i
$$

### 多分类逻辑回归

对二分类进行延伸，当$y_i$标记为c时，我们令$p_i$等于在c位置为1的独热编码。
$$
\theta_{i,c} = P(y=c|x_i) = \frac{exp(x_i^Tw_c)} {\sum_{c'=1}^C exp(x_i^Tw_{c'})} \\\\
\hat y = argmax_y P(y|x)
$$

假设我们想比较数据点更有可能两类中哪一类。
$$
 \frac{P(y=c_1|x_i)} {P(y=c_2|x_i)} 
 = \frac{exp(x_i^Tw_{c_1})} {exp(x_i^Tw_{c_2})} = 1 \\\\
 x_i^T (w_{c_1} - w_{c_2}) = 0
$$
假设iid。
$$
p_w(y_1, ..., y_n) = \prod_{i=1}^n \prod_{c=1}^C \theta_{i,c}^{I\{y_i=c\}}
$$
我们对其取对数
$$
log(p_w) = -\sum_i H(p_i, q_i) \\\\
= \sum_{i=1}^n \sum_{c=1}^C I\{y_i=c\} log(\frac{exp(x_i^Tw_c)} {\sum_{c'=1}^C exp(x_i^Tw_{c'})}) \\\\
= \sum_{i=1}^n \sum_{c=1}^C I\{y_i=c\}[x_i^Tw_c -log(\sum_{c'=1}^C exp(x_i^Tw_{c'}))] \\\\
= \sum_{i=1}^n [x_i^Tw_{y_i} - log(\sum_{c'=1}^C exp(x_i^Tw_{c'}))] \\\\
$$
这个函数没有解析解，只能使用梯度下降找到最值。
$$
\nabla_{w_c} f(w) = \sum_i (I\{y_i=c\} - \frac{exp(x_i^Tw_c)} {\sum_{c'=1}^C exp(x_i^Tw_{c'})}) x_i
$$

### 思路延伸

我们可以从另一个角度来理解。二分类逻辑回归相当于一个神经元，而多分类逻辑回归相当于神经网络中一层神经元，只不过激活函数是sigmoid函数。当多层神经元进行叠加，我们的目标函数不一定是凸函数了。不过，我们的决策边界也从线性边界拓宽到非线性边界。那么我们可以将这个神经网络用于回归问题。



# 神经网络

### 损失函数

$$
min_{W_1,W_2,...} \sum_i l(y_i, y_i) \\\\
min_{W_1,W_2,...} \sum_i D(y_i||\phi(W_l^T...\phi(W_2^T\phi(W_1^Tx_i))...))\\\\
$$

此处的损失函数沿用了K-L散度而非交叉熵（见逻辑回归章节）。这已经不是凸优化问题了，我们只能使用SGD进行求解。

### 训练过程

如果我们的激活函数$\phi$是Sigmoid，那么
$$
\sigma(z) = \frac 1 {1+e^{-z}} \\\\
\frac{d\sigma(z)}{dz} = \sigma(z)(1-\sigma(z))
$$
使用反向传播(backpropagation)计算梯度：向网络输入数据$x_i$，计算其相对于前一层网络权重的梯度。

### 函数逼近

英文是function approximation。

#### 通用近似定理

英文是universal approximation theorem。人工神经网络近似任意函数的能力。

考虑一个单层NN类似$\hat y = f(x) = \phi(W_2^T\phi(W_1^Tx))$。对于该映射函数$f: R^d \rightarrow R^D$，一定存在一组权重$W_1, W_2$满足
$$
sup_x||y - f(x)|| \le \epsilon
$$
此处$\epsilon$是错误率，非负。$sup$代表上界，与$max$的区别是最大值必须在值域内（函数值必须可以取到max），但是$sup$可以不在值域内。当$max$存在的时候，两者相等。



# 自编码器

### 引子：伯努利实验的观察

在n次伯努利试验下，根据霍夫丁不等式，我们可以推出
$$
P(|k - n\theta| \ge m) \le 2*exp(-2m^2/n)
$$
现在有一种彩票，序列是100次伯努利实验结果的拼接。$X \sim Bern(0.9)$。

假设我们赢得彩票的方式是猜对编码。假设我们只买一张，那么我们应该买序列全是1的彩票，因为其概率最大。假设我们可以多买100张彩票，那么我们应该将序列中包含1个0的彩票也全部买下。购买顺序以此类推。

假设我们赢得彩票的方式是猜对中奖序列中1的个数。那我们应该猜90。

<img src="https://i.postimg.cc/4dDZZNH3/coin-binomial.png" height=280>

如果对信息熵进行绘图，我们会发现他的形状会和第一张图C(n,k)的形状一模一样。

### 构造

<img src="https://i.postimg.cc/jdGmmgmQ/autoencoder.png" height=200>

自编码器(autoencoder)由编码器和解码器两部分组成。编码器负责压缩，解码器负责还原。我们的目标是使得编码的表示更加高效，即在尽可能减小信息损失的前提下，中间产物$y$的大小应该远小于输入$x$。

### 信源编码定理

Shannon的信源编码定理(source coding theorem)是量化信息的基础。我们考虑离散且有限的随机变量$X$。n个iid随机变量，每个变量的信息熵都是$H(X)$。这组变量可以压缩为$nH(X)$ bits的大小，而不用太担心有过多的信息损失。换言之，如果压缩的大小小于这个临界点，那么一定会有信息损失。

接下来是进一步解释该定理，不过不是严格的证明。

我们假设encoder的输入$x \in \{0, 1\}^n$并将其映射到$y \in \{0, 1\}^m$。数据是$X \sim Bern(0.8)$。

我们先看$n = 1000$的情况。假设我们自编码器的波动容纳是100，也就是说当$x$中包含少于700个1，或者多于900个1的时候自编码器会失效。我们下面计算自编码器失效的概率。
$$
P(|k-n\theta| \ge t) \le 2*exp(-2t^2/n) \\\\
P(|k-800| \ge 100) \le 2*exp(-2*100^2/1000) = 4.12*10^{-9} \\\\
$$
在之前，我们需要对所有序列编码。
$$
\sum_{k=1}^{1000} C(1000, k) = 2^{1000}
$$
我们现在只需要对规定范围内的序列进行编码。
$$
\sum_{k=700}^{900} C(1000, k) \approx 2^{877}
$$
目前的压缩率是$877/1000 = 0.877$。

<img src="https://i.postimg.cc/4y4ScDcg/source-coding-1000.png" height=240>

改变出现在指数项是量级上的改变！

我们再来看$n = 100,000$的情况。假设我们自编码器的波动容纳是1000，也就是说当$x$中包含少于79000个1，或者多于81000个1的时候自编码器会失效。我们下面计算自编码器失效的概率。
$$
P(|k-n\theta| \ge t) \le 2*exp(-2t^2/n) \\\\
P(|k-80000| \ge 1000) \le 2*exp(-2*1000^2/100000) = 4.12*10^{-9} \\\\
$$
我们需要编码的序列范围是
$$
\sum_{k=79000}^{81000} C(100000, k) \le 2000*C(100000,79000) \approx 2^{74163}
$$
目前的压缩率是$74163/100k = 0.74163$。序列的信息熵是0.722 bits。已经非常接近了。

我们总结一下，当$n$足够大，始终保持自编码器的波动容纳是$t = c \sqrt n$，那么自编码器失效的概率是
$$
P(|k-n\theta| \ge t) \le 2*exp(-2t^2/n) = 2*exp(-2c^2)
$$
我们需要编码的序列是
$$
\sum_{k=n\theta-t}^{n\theta+t} C(n, k) \\\\
\approx 2t*C(n, n\theta) \\\\
\approx 2^{log_2(2t)} 2^{nH(X)}
$$
由此得知，压缩率大约是$H(X)$。

已知斯特灵公式(Stirling's approximation)
$$
n! = \sqrt{2\pi n}*(n/e)^n \approx n^n
$$
我们来看看上式后半部分成立的原因。
$$
log[C(n, n\theta)] = log[\frac{n!} {k!(n-k)!}] \\\\
\approx log[\frac{n^n} {k^k(n-k)^{(n-k)}}] \\\\
= nlog(n) - klog(k) - (n-k)log(n-k) \\\\
= (n-k)log(n) + klog(n) - klog(k) - (n-k)log(n-k) \\\\
= (n-k)log(\frac{n} {n-k}) + k log(\frac{n} {k})
$$
将$k = \theta n$代入上式，有
$$
log[C(n, \theta n)] = n\theta log(\frac{1} {\theta}) + n(1-\theta)log(\frac{1} {1-\theta})
$$

### 渐近等分性质

在信息论中，渐近均分性质(Asymptotic equipartition property, AEP)是随机源输出样本的一般性质。对于数据压缩理论中使用的典型集合的概念而言，这是基础。

我们只关心熵典型集(entropy typical set)
$$
A_{\epsilon}^n = \{x:|\frac 1 n log(\frac 1 {p(x)}) - H(X)| \le \epsilon \}
$$
重写熵典型集的定义（将绝对值拆开），我们可以得到，在$x \in \mathcal X^n$时，如下不等式成立。
$$
2^{-n(H(X)+\epsilon)} \le p(x) \le 2^{-n(H(X)-\epsilon)}
$$
这个不等式在对性质的证明上很有帮助。

**性质**

对于$n$足够大时，

1. $P(A_{\epsilon}^n) \ge 1-\epsilon$
2. $|A_{\epsilon}^n| \le 2^{n(H(X)+\epsilon)}$

3. $|A_{\epsilon}^n| \ge (1-\epsilon) 2^{n(H(X)-\epsilon)}$

对于性质2，有如下证明
$$
1 = \sum_{x \in \mathcal X} p(x) \\\\
\ge \sum_{x \in A_{\epsilon}^n} p(x) \\\\
\ge \sum_{x \in A_{\epsilon}^n} 2^{-n(H(X)+\epsilon)} \\\\
= 2^{-n(H(X)+\epsilon)}|A_{\epsilon}^n| \\\\
$$
可得$|A_{\epsilon}^n| \le 2^{n(H(X)+\epsilon)}$

对于性质3，有如下证明
$$
1-\epsilon < P(x \in A_{\epsilon}^n) \\\\
\le \sum_{x \in A_{\epsilon}^n} 2^{-n(H(X)-\epsilon)} \\\\
= 2^{-n(H(X)-\epsilon)}|A_{\epsilon}^n| \\\\
$$

可得$|A_{\epsilon}^n| \ge (1-\epsilon) 2^{n(H(X)-\epsilon)}$

### 损失函数

假设我们的激活函数是线性函数，我们的损失函数是MSE。中间产物的维度是$m$。
$$
\hat x = W_2^TW_1^Tx \\\\
W_2^TW_1^T \approx I \\\\
rank(W_2^TW_1^T) \le m < d
$$
我们现在计算它的损失。我们令$A = W_2^TW_1^T$。
$$
min_{W_1, W_2} \sum_i l(x_i, \hat x_i) \\\\
= min_{W_1, W_2} \sum_i ||x_i - W_2^TW_1^Tx_i||_2^2 \\\\
= min_{W_1, W_2} ||X - W_2^TW_1^TX||_2^2 \\\\
= min_{A:rank(A)\le m} ||X - AX||_F^2
$$

### 应用

自编码器的应用体现在mp3和jpeg形式上，而gzip利用的是霍夫曼编码(Huffman coding)。

可以用于去除噪音、数据可视化、流形学习(manifold learning)等领域。



# 变分自动编码器

是自编码器的升级版，目标是构建一个从隐变量$Z$生成目标数据$X$的模型。我们需要假设$Z$服从某种分布，最常见的就是正态分布（也有假设服从均匀分布的）。接下来，我们只需要保证模型能够学习到分布的参数即可。这个就是变分自动编码器和自动编码器的最主要的区别，变分自动编码器只学习参数，而自动编码器学习代表数据的函数。

我们以MNIST数据集为例展示过程如下。

<img src="https://i.postimg.cc/Y0VzxYx3/VAE.png" height=200>

我们延续之前自编码器的符号标记，用$f$代表编码器部分的函数，用$g$代表解码器部分的函数。这两个函数之间有点类似互为反函数的关系。我们的目标就是使得$\hat x \approx x$，需要令$p(z) \sim \mathcal N(0,I)$。后一个要求是为了计算简便，方便根据学习到的参数进行抽样。

我们再来复习一下隐变量模型。该模型的目的是从数据集$D = \{(x_i)\}^n$中学习$x$的分布$p(x)$。其中$Z$是隐变量，有可能有具体的意义，也有可能只是为了计算的方便。
$$
p(x) = \sum_z p(x,z) = \sum_z p(z) p(x|z) \\\\
p(x) = \int p(x,z) dz = \int p(z) p(x|z) dz
$$
在MNIST数据集中，$Z$是预期数字。

我们先考虑$Z$是离散变量的情况，即预测$x$分布的模型是混合模型。
$$
p(x) = \sum_{k=1}^K p(Z=k) \mathcal N(x|\mu_k, \Sigma_k)
$$
我们希望学习到$x$和$z$之间的关系，也就是后验概率。
$$
p(z|x) = \frac{p(x,z)} {p(x)} = \frac{p(x|z)p(z)} {p(x)} = \frac{p(x|z)p(z)} {\sum_z p(x|z)}
$$
例如，在之前介绍的高斯混合模型使用EM算法，我们知道
$$
p(z|x) = r_{i,k} = \frac{\pi_k \mathcal N(x| \mu_k, \Sigma_k)} {\sum_{k'} \pi_k' \mathcal N(x| \mu_k', \Sigma_k')}
$$
变分推理(variational inference)：我们希望通过另一个分布$q(z|x)$来估计$p(z|x)$。我们将$q(z|x)$限制成容易处理的形式，例如正态分布$\mathcal N(\mu(x), \Sigma(x))$。我们用K-L散度来计算两个分布之间的相似程度。为了书写简便，我们将$q(z|x)$简写成$q(z)$。
$$
D(q(z) || p(z|x)) = \int q(z)log \frac{q(z)} {p(z|x)} dz \\\\
= \int q(z)log \frac{q(z)p(x)} {p(x,z)} dz \\\\
= \int q(z) [log \frac{q(z)} {p(x,z)} + log(p(x))] dz \\\\
= \int q(z) log \frac{q(z)} {p(x,z)} dz + \int q(z) log(p(x)) dz \\\\
= \int q(z) log \frac{q(z)} {p(x,z)} dz + log(p(x))
$$
我们知道K-L散度具有非负性，所以我们可以改写上式成下面形式。
$$
log(p(x)) \ge \int q(z) log \frac{p(x,z)}{q(z)} dz \\\\
= \int q(z) log \frac{p(x|z)p(z)}{q(z)} dz \\\\
= \int q(z) log(p(x|z)) dz + \int q(z) log \frac{p(z)}{q(z)} dz \\\\
= E_{Z \sim q(z)}[log(p(x|z))] - D(q(z)||p(z))
$$
我们的目标是最大化$p(x)$，实际上就等同于最大化下界中的期望，最小化下界中的K-L散度。我们之前求解过类似的K-L散度。
$$
D(p(x) || q(x)) = \frac 1 2 log(\frac{|\Sigma_2|} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_2^{-1}\Sigma_1) + (\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2) - \frac{n} {2} \\\\
$$
这里，我们将$p(z) \sim \mathcal N(0,I)$和$q(z) \sim \mathcal N(\mu(x),\Sigma(x))$代入。
$$
\frac 1 2 log(\frac{1} {|\Sigma_1|}) + \frac{1} {2}tr(\Sigma_1(x)) + (\mu(x))^2 - \frac{n} {2} \\\\
$$
接下来我们要计算目标函数前面的部分。
$$
E_{Z \sim q(z)}[log(p(x|z))] \approx \frac 1 n \sum log(p(x_i|z))
$$
其中$g$是确定性函数，所有的不确定性都来自根据分布的参数进行抽样。所以，如果我们知道了$z$，我们就知道了$\hat x$，反之亦然。
$$
log p(x_i|z) = log p(x_i|\hat x_i)
$$
再用那张图复习一遍。

<img src="https://i.postimg.cc/g2KSpZZK/VAE2.png" height=100>

我们想要最大化这个期望，那么就需要使$\hat x$和$x$尽可能接近。在损失函数的选择上，平方误差函数(squared error loss)可以帮我们最大化。这里直接给出损失函数公式。
$$
l(x,\hat x) = ||x -\hat x||^2 + D(\mathcal N(\mu, \Sigma)||\mathcal N(0,I))
$$





# Reference

- Probability and Information Theory in Machine Learning (ECE 601), Matthew Malloy, Fall 2020
- Machine Learning, A Probabilistic Perspective, Kevin P. Murphy, 2012



