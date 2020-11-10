# 主成分分析技术

全称是Principal component analysis (PCA)。将原始数据从p个特征维度降低到d个维度。

<img src = "./pics/pca.png" height=300>

## 原理

对原始特征空间进行重构。需要最大投影方差，尽可能保留数据在原空间的信息。投影就是$x^Tw$，我们必须规定$|w| = 1$，否则可以通过增大$w$来增大投影方差，失去了意义。具体可以参见损失函数部分。

从另外一个角度理解，就是需要最小化重构数据和原数据之间的距离。

#### 输入

训练集数据$D = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^p$，$y_i \in  R$

$X = (x_1\ x_2\ ...\ x_M)^T \in R^{M*p}$

同时，我们可以得到数据均值和方差。让我们用$1_M$表示一个长度为M，值全为1的列向量。用$H$表示中心矩阵。$H = I_M - \frac 1 M 1_M 1_M^T$。通过计算我们可以得到$H^n = H$。

$\bar x = \frac{1}{M} \sum_{i=1}^M x_i = \frac{1}{M} X^T 1_M$

$S = \frac{1}{M} \sum_{i=1}^M (x_i - \bar x)^T(x_i - \bar x) = \frac{1}{M} X^T H X$

#### 输出

$X' = (x_1'\ x_2'\ ...\ x_M')^T \in R^{M*d}$

#### 损失函数

因为数据是$p$维的，我们让$u_i$当作每一个维度上的$w$，仍然满足$|u_i| = 1$。我们的目标是根据投影方差定义的。
$$
J = \frac 1 M \sum_{i=1}^M ((x_i - \bar x)u_1)^2 \\\\
= \frac 1 M \sum_{i=1}^M u_1^T (x_i - \bar x) (x_i - \bar x)^T u_1\\\\
= u_1^T \frac 1 M \sum_{i=1}^M  (x_i - \bar x) (x_i - \bar x)^T u_1\\\\
= u_1^T S u_1
$$
那么我们可以得到$u_1^* = argmax_u\ u_1^T S u_1$且满足$|u_i| = 1$。

为了体现出$|u_i| = 1$的特点，我们对损失函数进行重构。

$L(u_1, \lambda) = u_1^T S u_1 + \lambda(1 - u_1^T u_1)$

接下来对$u_i$进行求导，
$$
\frac {\partial L} {\partial u_1} = 2Su_1 - 2\lambda u_1 = 0 \\\\
Su_1 = \lambda u_1
$$
这里我们可以发现$u_1$是S的eigen vector，$\lambda$是对应的eigen value。

其他$u_i$通过相同过程求解，只需要满足$u_i$之间是互相垂直的即可。

### 步骤

（1）将样本矩阵中心化得到$X'$，即每一维度减去该维度的均值，使维度上的均值为0

（2）计算样本协方差矩阵$S = \frac{1}{M-1} X^{'T}X'$

（3）寻找协方差矩阵的eigen values和eigen vectors

降维之后对原始数据进行还原

$\hat X = \sum_{j=1}^d (X'^T u_j)u_j$

## 适用场景

原始数据特征过于庞大且特征有明显的相关性。

**优点**

1. 计算简单，容易实现
2. 在一定程度上起到降噪效果
3. 无参数限制

**缺点**

1. 降维之后的数据缺乏解释性







# Reference

- CS540 Intro to AI, UW Madison, Jerry Zhu
- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007