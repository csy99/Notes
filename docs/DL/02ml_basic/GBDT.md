# 梯度提升决策树

英文是Gradient Boosting Decision Tree (GBDT)。是一种迭代的决策树算法，由多棵决策树组成，将所有树的结论累加起来做最终答案。值得注意的是，GBDT中的树都是回归树，不是分类树。GBDT主要用于回归问题（包括线性和非线性），但也可以用于分类问题。

我们在已经搞清楚决策树(DT)部分的前提下，可以简单看看梯度迭代(GB)。GB核心的思想就是每一棵树都在学习之前所有树结论和的残差。

### 输入

训练集数据$D = \{(x_i,y_i)\}_{i=1}^N$。损失函数$L(y, f(x))$。我们令树的数量为$M$，用$m$表示当前树的序号。

### 步骤

1. 初始化$f_0(x) = argmin_\gamma \sum_{i=1}^N L(y_i, \gamma)$

2. 对于$m = 1 \sim M$:

   a) 对于每一个观测$i = 1 \sim N$计算伪残差(pseudo-residuals)

$$
r_{im} = -[\frac{\partial L(y_i, f_{m-1}(x_i))}{\partial f_{m-1}(x_i)}] \\\\
$$

​		b) 对数据集$\{(x_i, r_{im})_{i=1}^N\}$拟合一棵回归树$h_m(x)$来学习残差。

​		c) 对于一维优化问题，计算
$$
\gamma_{m} = argmin_\gamma \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + \gamma h_m(x_i))
$$
​		d) 更新当前树的函数
$$
f_m(x) = f_{m-1}(x) + \gamma_{m} h_m(x)
$$

3. 输出最终的函数$\hat y = f_M(x)$

解释如下：

1、初始化，估计使损失函数极小化的常数值，它是只有一个根节点的树，即$\gamma$是一个常数值。使用常数值的原因是常数是拟合数据集最差的模型（直接假设$x$和$y$直接没有关系），模型可以通过不断迭代找到更好的模型。

2、
 （a）计算损失函数的负梯度在当前模型的值，将它作为残差的估计
 （b）估计回归树叶节点区域，以拟合残差的近似值
 （c）利用线性搜索估计叶节点区域的值，使损失函数极小化
 （d）更新回归树

3、得到输出的最终模型 $f(x)$。

### 场景

近年来多被用于搜索排序。

**优点**

1. 通过迭代学习残差，提高模型精确度

**缺点**

1. 耗时较长，模型无法并行运行
2. 超参数$M$即决策树的数量非常重要，不好调整
3. 容易出现过拟合



# Xgboost

英文是eXtreme Gradient Boosting (XGBoost)。针对传统GBDT算法做了很多细节改进，包括损失函数、正则化、切分点查找算法优化、稀疏感知算法、并行化算法设计等等。

### 输入

训练集数据$D = {(x_i,y_i)}_{i=1}^N$。损失函数$L(y, f(x))$。我们令树的数量为$M$，用$m$表示当前树的序号。

限定条件为，
$$
f(x) = \sum_{j=1}^M h_j(x) \\\\
f(x)_m = \sum_{j=1}^m h_j(x) \\\\
$$

### 目标函数

$J = \sum_{i=1}^N L(y_i, f(x_i)) + \sum_{m=1}^M \Omega(f_m)$。

前一部分就是普通的损失函数，一般是MSE或者Logistic Loss。后一部分是惩罚函数，用于控制树的复杂度，帮助抑制过拟合。

### 步骤

我们浅析其计算过程。
$$
\hat y_i^{(0)} = 0 \\\\
\hat y_i^{(1)} = f_1(x_i) = \hat y_i^{(0)} + f_1(x_i) \\\\
\hat y_i^{(0)} = f_1(x_i) + f_2(x_i) = \hat y_i^{(1)} + f_2(x_i) \\\\
...\\\\
\hat y_i^{(t)} = \sum_{k=1}^m f_k(x_i) = \hat y_i^{(m-1)} + f_m(x_i) \\\\
$$
由于这是个树模型，传统SGD不太适用。所以我们使用增量训练(additive training)。我们需要对目标函数进行泰勒展开，这要求损失函数至少二阶可导。这里不详细介绍。

### 重要参数

根据经验，介绍模型中最需要调整的参数。

>常规参数General Parameters
>
>booster: 默认是gbtree；也可以Dart模式，类似Dropout的思想
>
>树参数Tree Parameters
>
>max_depth：树的深度，最重要的参数，过深的话容易过拟合。
>
>eta: 学习率learning_rate，0.01~0.2
>
>gamma：在树的叶子节点上进行进一步分区所需的最小损失减少。利用CV进行微调
>
>min_child_weight: 每一支需要的观测实例的最小权重(hessian)，利用CV进行微调
>
>学习任务参数Learning Task Parameters
>
>eval_metric：评判标准，根据数据的不同进行设置

### 场景

**优点**

1. 不需要过多特征工程
2. 能够得到特征重要性
3. 鲁棒性强，较少受到异常点(outliers)的影响
4. 速度快

**缺点**

1. 难以可视化
2. 模型参数多，难以进行微调



# Reference

- NLP实战高手课，第三章，王然，极客时间
- [GBDT: 梯度提升决策树](https://www.jianshu.com/p/005a4e6ac775)，Siyue Lin，简书
- [Demystifying Maths of Gradient Boosting](https://towardsdatascience.com/demystifying-maths-of-gradient-boosting-bd5715e82b7c), Krishna Kumar Mahto, towards data science

