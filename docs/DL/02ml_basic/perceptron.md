# 感知机

## 原理

思想是错误驱动。一开始赋予w一个初始值，通过计算被错误分类的样本不断移动分类边界。

#### 输入

训练集数据$D = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^p$，$y_i \in  \{-1, +1\}$

$X = (x_1\ x_2\ ...\ x_M)^T \in R^{M*p}$

$f(x) = sign(w^Tx+b)$

#### 输出

{+1, -1}

#### 损失函数

可以采用0-1 loss。

$L(w) = \sum^M_{i=1} I\{y_iw^Tx_i < 0\}$

但是这个损失函数不可导。我们采用它的变形形式。

$L(w) = \sum^M_{i=1} -y_iw^Tx_i$

这样，就可以采用SGD进行求解。
$$
w^{t+1} = w^t - \eta*(\nabla_w L(w)) \\\\
= w^t + \eta y_ix_i
$$
如果不收敛，可以采用pocket algorithm。

## 适用场景

数据集需要线性可分。

**优点**

1. 模型计算简单，建模速度快
2. 可解释性好

**缺点**

1. 需要数据线性可分





# Reference

- 《美团机器学习实践》by美团算法团队，第三章
- 《机器学习》by周志华，第三、四章
- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007