# 支持向量机

全称Support Vector Machine (SVM)。可以分为硬间隔（hard margin SVM），软间隔（soft margin SVM），和核支持向量机（kernel margin SVM）。

## 原理

### 输入

训练集数据$D = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^p$，$y_i \in  R$

$X = (x_1\ x_2\ ...\ x_M)^T \in R^{M*p}$

$f(x) = sign(w^Tx+b)$

正则化参数$\lambda_1$ ，$\lambda_2$

### 输出

线性回归模型$\hat f(x)$

### 损失函数

硬间隔SVM也称为最大间隔分类器。

$margin(w, b) = min\ distance(w, b, x_i)$

为了简化运算，我们指定最小的margin为1（可以通过缩放实现）。我们希望达成以下目标。

$max\ margin(w, b)\ s.t.\ y_i(w^Tx_i+b) > 0 \ \forall \ i =1 \sim M$

进行数学推导，
$$
max\ margin(w, b) = max_{w, b} min_{x_i} \frac{1} {||w||} y_i(w^T x_i + b) \\\\
= max_{w, b} \frac{1} {||w||} min_{x_i} y_i(w^T x_i + b) \\\\
$$
可以简化成
$$
max \ margin(w, b) = max_{w, b} \frac{1} {||w||} \\\\
s.t. \ y_i(w^Tx_i + b) >= 1 \  \forall \ i =1 \sim M
$$
或者
$$
max \ margin(w, b) = min_{w, b} {||w||} \\\\
s.t. \ y_i(w^Tx_i + b) >= 1 \ \forall \ i =1 \sim M
$$
从而我们很容易得到损失函数（$\lambda \ge 0$），
$$
L(w, b, \lambda) = \frac 1 2 w^Tw + \sum_{i=1}^M \lambda_i(1 - y_i(w^Tx_i+b)) \\\\
min_{w, b} \ max_{\lambda} L(w,b,\lambda) \ s.t. \ \lambda_i \ge 0
$$
当$y_i(w^Tx_i+b) > 0$时，很容易证明$L$的最大值是正无穷。当$y_i(w^Tx_i+b) \le 0$时，$L$的最大值是在$\lambda$取0时，整体等于$\frac 1 2 w^Tw $。

我们可以证明$min \ max \ L = max \ min \ L$，此处省略。这个被称为强对偶关系。

我们将上面的原问题转化成对偶问题如下，
$$
max_{\lambda}\ min_{w, b} \  L(w,b,\lambda) \ s.t. \ \lambda_i \ge 0
$$
进行求导，
$$
min_{w, b} \  L(w,b,\lambda) \\\\
\frac{\partial L} {\partial b} = \frac{\partial [\sum_{i=1}^M \lambda_i - \sum_{i=1}^M \lambda_i y_i(w^Tx_i+b)]} {\partial b} = 0\\\\
解得\sum_{i=1}^M \lambda_i y_i= 0 \\\\
将其代入 L \\\\
L(w, b, \lambda) = \frac 1 2 w^Tw + \sum_{i=1}^M \lambda_i - \sum_{i=1}^M \lambda_i y_i w^T x_i \\\\
\frac{\partial L} {\partial w} = 0 \\\\
w^* = \sum_{i=1}^M \lambda_i y_i w^T x_i \\\\
将其代入 L \\\\
L(w, b, \lambda) \\\\
= \frac 1 2 (\sum_{j=1}^M \lambda_j y_j w^T x_j)^T(\sum_{j=1}^M \lambda_j y_j w^T x_j) + \sum_{i=1}^M \lambda_i - \sum_{i=1}^M \lambda_i y_i (\sum_{j=1}^M \lambda_j y_j w^T x_j)^T x_i \\\\
= -\frac 1 2 (\sum_{i=1}^M \sum_{j=1}^M \lambda_i \lambda_j y_i y_j x_i x_j) + \sum_{i=1}^M \lambda_i
$$

从而，我们将问题再次进行了转换，我的目标是
$$
min_{\lambda} \  \frac 1 2 (\sum_{i=1}^M \sum_{j=1}^M \lambda_i \lambda_j y_i y_j x_i x_j) - \sum_{i=1}^M \lambda_i \\\\
\lambda_i \ge 0 \\\\
\sum_{i=1}^M \lambda_i y_i = 0
$$
因为我们的原问题和对偶问题具有强对偶关系，我们通过KKT条件
$$
\frac {\partial L} {\partial w} = 0  \\\\
\frac {\partial L} {\partial b} = 0  \\\\
\frac {\partial L} {\partial \lambda} = 0 \\\\
\lambda_i(1-y_i(w^Tx_i+b)) = 0 \\\\
\lambda_i \ge 0 \\\\
1-y_i(w^Tx_i+b) \le 0
$$
可以得到最优解，

$w^* = \sum_{i=1}^M \lambda_i y_i x_i$

我们还需要代入一个处于边界上的点$(x_k, y_k)$满足$ 1-y_k(w^Tx_k+b) = 0$，再求解偏置
$$
1 - y_k(w^Tx_k+b) = 0 \\\\
y_k(w^Tx_k+b) = 1 \\\\
y_k^2(w^Tx_k+b) = y_k \\\\
(w^Tx_k+b) = y_k \\\\
b^* = y_k - w^Tx_k \\\\
b^* = y_k - \sum_{i=1}^M \lambda_i y_i x_i^T x_k
$$
软间隔SVM允许少量错误。

$L(w, b) = min \frac 1 2 w^Tw + loss$

我们可以将后面额外的损失定义为0-1 loss，更常用的是hinge loss。

那么，我们可以重新定义损失函数为
$$
L(w,b) = min_{w,b} \frac 1 2 w^Tw + C*\sum_{i=1}^M max(0, 1-y_i(w^Tx_i+b)) \\\\
1-y_i(w^Tx_i+b) \le 0
$$
$C$起到了一个正则化的作用。

## 适用场景

普遍适用。

**优点**

1. 边界只由少数的支持向量所决定，避免维度灾难
2. 可解释性好
3. 对离群数据敏感度较小，鲁棒性高

**缺点**

1. 对大规模训练样本而言，消耗大量内存和运算时间
2. 解决多分类问题时，需要多组SVM模型



# 核方法

对应英文是Kernel Method。核方法用于解决有数据集类别之间的边界压根不是线性的。对于原始的输入空间$\mathcal X$，使用$\phi(x)$进行非线性转换成为特征空间$\mathcal Z$，从而达到线性可分的状态。理论基础是Cover Theorem，即高维空间比低维空间更易线性可分。

在$\phi(x)$维度非常高的情况下，求$\phi(x_i)$非常困难。我们发现有一种核技巧(kernel trick)，可以在不需要单独计算$\phi(x_i)$和$\phi(x_j)$的前提下得到$\phi(x_i)^T\phi(x_j)$。毕竟后者才是我们$L$中需要得到的值。

$K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$

一般情况下，我们的核函数$K$指的是正定核函数。有函数$K$可以做到从$\mathcal X * \mathcal X$到$R$的映射，$\exist \Phi \in \mathcal H$,使得$K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$，那么称$K$为正定核函数。我们再介绍一下希尔伯特空间$\mathcal H$。它是完备的（对极限是封闭的），可能是无限维的，被赋予内积运算的一个线性空间。

**正定核函数性质**

1. 对称性，即$K(x_i, x_j) = K(x_j, x_i)$
2. 正定性，即任取$\mathcal X$中的M个元素，从$x_1$到$x_M$，对应的Gram matrix是半正定的

我们来证明正定核函数的性质。先证明必要性。

已知$K(x_i, x_j) = <\phi(x_i),\phi(x_j)>$，要证明其对称性和正定性。

对称性可以由内积的对称性证明。我们现在想要证明对应的Gram matrix是半正定的。令Gram matrix是$G = [K(x_i, x_j)]$。所以需要证明对于任意$\alpha \in R^M$，有$\alpha^T G \alpha \ge 0$。
$$
\alpha^T G \alpha \\\\
= \sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j K(x_i, x_j) \\\\
= \sum_{i=1}^M \sum_{j=1}^M \alpha_i \alpha_j \phi(x_i)^T\phi(x_j) \\\\
= \sum_{i=1}^M \alpha_i \phi(x_i)^T \sum_{j=1}^M \alpha_j \phi(x_j) \\\\
= [\sum_{i=1}^M \alpha_i \phi(x_i)]^T [\sum_{j=1}^M \alpha_j \phi(x_j)] \\\\
= ||\sum_{i=1}^M \alpha_i \phi(x_i)||^2 \ge 0
$$
我们就证明了Gram Matrix是半正定的。

# 约束优化

我们定义原问题的最优解为$d^*$，对偶问题的最优解是$p^*$。

定义原问题$min_x f(x)$，有N个不等式约束，$n_i(x) \le 0$，有M个等式约束，$m_i(x) = 0$

转换后的原问题的无约束形式为$min_x \ max_{\lambda, \eta} = L(x, \lambda, \eta)$，$\lambda_i \ge 0$

下面是转换的说明。

拉格朗日函数：

$L(x, \lambda, \eta) = f(x) + \sum_{j=1}^N \lambda_j n_j + \sum_{i=1}^M m_i \eta_i$

如果$x$违反了不等式约束，那么$max_\lambda L$一定会趋近于正无穷。所以在其前面加上一个$min$相当于进行了一次过滤，将所有不满足不等式约束的$x$都过滤掉了。

### 弱对偶

我们接着证明原问题和对偶问题是相等的。对偶问题是$max_x \ min_{\lambda, \eta} = L(x, \lambda, \eta)$，$\lambda_i \ge 0$。

我们先证明弱对偶性，原问题的值会大于等于对偶问题，即$min \ max \ L \ge max \ min \ L$。
$$
min_x \ L(x, \lambda, \eta) \le L(x, \lambda, \eta) \le max_{\lambda, \eta} \ L(x, \lambda, \eta) \\\\
A(\lambda, \eta) \le B(x) \\\\
A(\lambda, \eta) \le min \ B(x) \\\\
max \ A(\lambda, \eta) \le min \ B(x) \\\\
$$

### Slater Condition

存在一点$x \in relint \ D$，使得对于所有的$n_i(x) < 0$。relint代表相对内部。

对于大多数凸优化问题，slater条件成立。

放松的slater条件是指，如果N中有K个仿射函数，那么只需要校验其余的函数满足slater条件即可。

通过弱对偶和Slater Condition可以推出强对偶关系。强对偶关系是下面的KKT条件的充要条件。

### 库恩塔克条件

通常被称为KKT条件。

#### 可行条件

有N个不等式约束，$n_i(x^*) \le 0$，有M个等式约束，$m_i(x^*) = 0$，$\lambda^* = 0$

#### 互补松弛条件

$\lambda_j^* n_j = 0$
$$
d^* = max_{\lambda, \eta} g(\lambda, \eta) \\\\
= g(\lambda^*, \eta^*) \\\\
= min_x L(x, \lambda^*, \eta^*) \\\\
= L(x^*, \lambda^*, \eta^*) \\\\
= f(x^*) + \sum_{j=1}^N \lambda_j^* n_j + \sum_{i=1}^M m_i^* \eta_i \\\\
= f(x^*)  + \sum_{j=1}^N \lambda_j^* n_j \\\\
= f(x^*) \\\\
= p^*
$$

#### 梯度为0

$$
min_x L(x, \lambda^*, \eta^*) \\\\
= L(x^*, \lambda^*, \eta^*) \\\\
$$



# Reference

- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007
- [支持向量机（SVM）的优缺点](https://blog.csdn.net/qq_38734403/article/details/80442535)