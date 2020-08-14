# 常用模型

## 3.1 线性回归

### 3.1.1 原理

#### 输入

训练集数据$T = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in  R$

$f(x) = w^Tx+b$

正则化参数$\lambda_1$ ，$\lambda_2$

#### 输出

线性回归模型$\hat f(x)$

#### 损失函数

$L = \sum^M_{i=1}(f(x_i) - y_i)^2$

均方误差对应欧氏距离。基于均方误差最小化求解的方法称为最小二乘法least square method。

$w^* = (X^TX)^{-1}X^Ty$ 此时$(X^TX)$是满秩矩阵。

我们也可以用模型逼近y的衍生物，比如$ln(y)$。在形式上仍是线性回归，不过实质上已是在求输入空间到输出空间的非线性函数映射。



## 3.2 逻辑回归

广义线性模型。

### 3.2.1 原理

#### 输入

训练集数据$T = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R^K$

损失函数$Cost(y,f(x))$

正则化参数$\lambda_1$ ，$\lambda_2$

学习速率$\alpha$，$\beta$

#### 输出

逻辑回归模型$\hat f(x)$

#### 判断函数

**单位阶跃函数**

不连续并且不充分光滑

**对数概率函数 Sigmoid**

$y = \frac{1}{1+e^{-z}}$    $z = w^Tx+b$

如果将y视为样本x作为正例的可能性，则1-y是反例可能性，两者的比值y/(1-y)称为几率，反映了x作为正例的相对可能性。上式是在用线性回归模型的预测结果去逼近真实标记的对数几率。

我们可以通过极大似然法来估计w和b。

$l(w,b) = \sum_{i=1}^M ln p(y_i | x_i;w,b)$

$P_w(y=j|x) = \frac{exp(x^Tw^{(j)})}{\sum_{k=1}^{K}exp(x^Tw^{(k)})}$

#### 损失函数

$cost = -log(\hat{p})$ if y =1. $cost = -log(1-\hat{p})$ if y =0. The reason why we are using log loss instead of MSE here is that it is a convex function. In addition, it will give larger updates when the error is larger. 



#### 算法3-1 逻辑回归算法

1. 随机初始化$\theta$
2. 计算$\theta_{j+1} = \theta_{j} - \alpha\frac{1}{m}\sum_{i=1}^{m}x_i[h(x_i)-y_i]$

3. 迭代

在迭代求解时使用高效的优化算法，如LBFGS、信赖域算法。这些求解方法是基于批量处理的，无法高效处理超大规模的数据集，也无法对线上模型进行快速实时更新。

随机梯度下降（SGD）是另一种优化方法，比如google的FTRL算法。



#### 算法3-2 FTRL算法

1. 对于$i\in\{i \sim d\}$，初始化$z_i = 0, n_i = 0$

2. 对样本t = 1 to T,

   1. 计算$$x_{t+1,i} = \left\{\begin{aligned} &0 ,&if |z_{t,i}| \leq \lambda_1 \\\ & -(\frac{\beta+\sqrt{n_i}}{\alpha}+\lambda_2)^{-1}(z_{t,i} - sign(z_{t,i})\lambda_1),& otherwise\end{aligned}\right.$$ 

   2. 计算$p_t = \sigma(x_tw)$，使用label函数和预测值$p_t$迭代

   3. 对于i

      $g_i = (p_t - y_t)x_i$

      $\sigma_i = \frac{1}{\alpha}(\sqrt{n_i + g_i^2} - \sqrt{n_i})$ 

      $z_i = z_i + g_i - \sigma_iw_{t,i}$

      $n_i = n_i +g_i^2$

3. 迭代

建议$\beta$取1。



## 3.3 线性判别分析 LDA

### 3.3.1 原理

给定训练样例集，将样例投影到一条直线上，使得同类样例的投影点接近，异类样例的投影点尽可能远离。LDA降维最多降到类别数k-1的维数。

#### 输入

训练集数据$T = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R^K$

$X_i, \mu_i, \Sigma_i$ 分别表示第i类样例的集合、均值向量、协方差矩阵。

#### 输出

每个分类的均值向量，各分类数据在总体中所占比例，降维矩阵，降维后各分量的权重。

#### 损失函数

定义全局散度矩阵$S_t = \sum_{i=1}^M(x_i-\mu)(x_i-\mu)^T$ 

定义类内散度矩阵$S_{w} = \sum_{k=1}^K S_{w_k}$。每个类的散度矩阵$S_{wk} = \sum_{x \in X_k}(x-\mu_k)(x-\mu_k)^T$ 

定义类间散度矩阵$S_b = S_t - S_w = \sum_{i=1}^K(\mu_i-\mu)(\mu_i-\mu)^T$

最大化目标，两个矩阵的广义瑞丽商$J = \frac {||w^T\mu_0 - w^T\mu_1||_2^2}{w^T\Sigma_0w+w^T\Sigma_1w} = \frac{w^TS_bw}{w^TS_ww}$

多分类优化目标 $max_W \frac{tr(w^TS_bw)}{tr(w^TS_ww)}$. tr表示矩阵的迹(trace)，是对角线元素总和。

分子分母只跟w的二次向有关，所以与w长度无关。

$min_w -w^TS_bw$ s. t. $w^TS_ww=1$  => $w = S_w^{-1}(\mu_0-\mu_1)$



## 3.4 多分类学习

### 一对一OvO

将K个类别两两配对，产生K(K-1)/2个二分类任务。存储开销和测试时间开销通常比OvR更大。

### 一对其余OvR

将一个类的样例作为正例，所有其他类的样例作为反例来训练N个分类器。在类别很多的时候，OvR的训练时间开销较大(每一次训练都是全量样本)。

### 多对多MvM

每次将若干个类作为正类，若干个其他类作为反类。

技术：纠错输出码ECOC。

过程：编码，对K个类别做p次划分，一共产生p个训练集，和p个分类器。解码，p个分类器分别对测试样本进行预测，预测标记组成一个编码。将编码与每个类别自己的编码比较，返回其中距离最小的类别。类别划分通过编码矩阵(二元码或者三元码)。

在测试阶段，ECOC编码对分类器的错误有一定的容忍和修正能力。一般来说，对同一个学习任务，编码越长，纠错能力越强(所训练的分类器越多)。



## 3.5 类别不平衡问题

基本策略就是再缩放。利用$\frac{y'}{1-y'} = \frac{y}{1-y}*\frac{m^-}{m^+}$。

### 欠采样

代表有EasyEnsemble算法。将反例划分成若干个集合供不同学习器使用，在全局来看不会丢失重要信息。

### 过采样

代表有Smote算法。

### 阈值移动

将基本策略内嵌。



## 3.6 决策树

```python
def TreeGenerate(D, A):
    生成节点node
    if D中样本全属于类别C:
        将node标记为C类叶节点
        return
    if A = ∅ or D中样本在A上取值相同:
        将node标记为D中样本数最多的类叶节点
        return       
    从A中选择最优划分属性a
    for a_v in a:
        为node生成一个分支;令Dv表示D中在a上取值为a_v的样本子集
        if Dv = ∅:
            将node标记为D中样本数最多的类叶节点
        else:
            以TreeGenerate(Dv, A\{a})为分支节点
```

三种终止递归条件：

1. 当前节点包含样本属于同一类，无需划分
2. 当前属性集合为空，或所有样本在所有属性上取值相同，无法划分
3. 当前节点包含的样本集合为空，不能划分

### 3.6.1划分选择

#### 信息增益 Info Gain

$p_k$ 是k类别在整体中所占比例。

信息熵 $Ent(D) = -\sum_{k=1}^K p_klog_2(p_k)$

信息熵的值越小，D的纯度越高。假设离散属性a有V个可能的取值，${a_1, a_2,..., a_V}$, 若使用a来对样本D进行划分，会产生V个分支节点。计算属性a对样本集D进行划分得到的信息增益。

$Gain(D, a) = Ent(D) - \sum_{v=1}^V \frac{D_v}{D} Ent(D_v)$

ID3决策树算法就是用信息增益为准则来选择划分属性。但是这种算法对可取值数目较多的属性有所偏好。

#### 增益率 Gain Ratio

$Gain_ratio(D, a) = \frac{Gain(D, a)}{IV(a)}$

其中$IV(a) = \sum_{v=1}^V \frac{|D_v|}{|D|}log_2\frac{|D_v|}{|D|}$被称为a的固有值。

C4.5决策树算法利用增益率来选择最优划分属性。

#### 基尼指数 Gini Index

Gini(D)反映从数据集D中随机抽取两样本，其类别标记不一致的概率。因此Gini(D)越小，数据集D的纯度越高。

Gini impurity $Gini(D) = 1 - \sum_{k=1}^K p_{k}^2$

属性a的基尼指数 Gini_index(D, a) = $\sum_{v=1}^V \frac{|D_v|}{|D|}Gini(D_v)$

选择使得划分后基尼指数最小的属性$a^* = arg min_{a \in A} Gini\_index(D, a)$

sklearn中使用的CART算法就是用Gini impurity做指标。

### 3.6.2 剪枝处理

缓和过拟合的主要手段。基本策略有预剪枝和后剪枝。

#### 预剪枝

在决策树生成过程中，若当前节点的划分不能带来泛化性能提升，则停止划分。预剪枝使得决策树的很多分支都没有展开，降低了过拟合风险，减少了训练和测试时间。不过，有些分支的当前划分虽不能提升泛化性能，但是在其基础上进行的后续划分却有可能导致性能显著提高，带来了欠拟合的风险。

#### 后剪枝

训练集生成的一棵完整决策树自下而上对节点进行考察，如果该节点对应子树替换为叶节点能带来决策树泛化性能提升，则将该子树替换为叶节点。欠拟合风险很小，不过训练时间要比不剪枝大得多。

### 3.6.3 连续值

连续属性离散化。最简单的策略就是采用二分法。给定样本集D和连续属性a，将该属性所有值排序。基于划分点t可将D分为$D^-_t$和$D^+_t$。之后我们可以计算信息增益来确定合适的分割点。

需要注意的是，与离散属性不同，若当前节点划分属性是连续属性，该属性还可作为其后代节点的划分属性。

### 3.6.4 缺失值处理

给定训练集D和属性a，用$\widetilde D$表示D中在属性a上没有缺失值的样本。为了解决在有属性值缺失的情况下进行属性划分，可以给每一个样本x赋予一个权重$w_x$。

$\rho = \frac{\sum_{x \in \widetilde D}w_x}{\sum_{x \in D}w_x}$表示无缺失值样本所占比例

$\widetilde p_k = \frac{\sum_{x \in \widetilde D_k}w_x}{\sum_{x \in \widetilde D}w_x}$表示无缺失值样本中第k类所占比例

$\widetilde r_v = \frac{\sum_{x \in \widetilde D_v}w_x}{\sum_{x \in \widetilde D}w_x}$表示无缺失值样本中在属性a上取值是$a_v$的样本所占比例

我们可以将信息增益的计算式推广成

$Gain(D, a) = \rho * Ent(\widetilde D) - \rho* \sum_{v=1}^V \widetilde r_v  Ent(\widetilde D_v)$

为了解决特定样本在该属性上的值缺失时的样本划分问题，让这个样本以不同的概率划入到不同的子节点中去。

### 3.6.5 多变量决策树

可以对分类边界进行不是沿着平行轴的方向进行划分的或者进行其他复杂划分的决策树。











# Reference

- 《美团机器学习实践》by美团算法团队，第三章
- 《机器学习》by周志华，第三、四章