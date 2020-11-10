# 逻辑回归

广义线性模型。

## 原理

### 输入

训练集数据$T = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R^K$，二分类$y_i \in  \{-1, +1\}$

损失函数$Cost(y,f(x))$

正则化参数$\lambda_1$ ，$\lambda_2$

学习速率$\alpha$，$\beta$

### 输出

逻辑回归模型$\hat f(x)$

### 判断函数候选

**单位阶跃函数**

不连续并且不充分光滑

**对数概率函数 Sigmoid**

$y = \frac{1}{1+e^{-z}}$    $z = w^Tx+b$

<img src="./pics/sigmoid.png" height="300">

如果将y视为样本x作为正例的可能性，则1-y是反例可能性，两者的比值y/(1-y)称为几率，反映了x作为正例的相对可能性。上式是在用线性回归模型的预测结果去逼近真实标记的对数几率。

我们可以通过极大似然法来估计w和b。

$l(w,b) = \sum_{i=1}^M ln p(y_i | x_i;w,b)$

$P_w(y=j|x) = \frac{exp(x^Tw^{(j)})}{\sum_{k=1}^{K}exp(x^Tw^{(k)})}$

### 损失函数

$cost = -ylog(\hat{p}) - (1-y)log(1-\hat{p})$. 

我们之所以使用对数概率函数而不是MSE的原因：(1)对数概率函数是一个凸函数；(2) 当误差较大时，对数概率函数可以提供较大的更新。

推导w的MLE。
$$
w^* = argmax_x P(Y|X) \\\\
= argmax_w \prod_{i=1}^{M} P(Y_i|x_i) \\\\
= argmax_w \sum_{i=1}^{M} log P(Y_i|x_i) \\\\
= argmax_w \sum_{i=1}^{M} [y_i log p_1 + (1-y_i) log p_0] \\\\
$$




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

$Gain\_ratio(D, a) = \frac{Gain(D, a)}{IV(a)}$

其中$IV(a) = \sum_{v=1}^V \frac{|D_v|}{|D|}log_2\frac{|D_v|}{|D|}$被称为a的固有值。

C4.5决策树算法利用增益率来选择最优划分属性。

#### 基尼指数 Gini Index

Gini(D)反映从数据集D中随机抽取两样本，其类别标记不一致的概率。因此Gini(D)越小，数据集D的纯度越高。

Gini impurity $Gini(D) = 1 - \sum_{k=1}^K p_{k}^2$

属性a的基尼指数 $ Gini\_index(D, a) = \sum_{v=1}^V \frac{|D_v|}{|D|}Gini(D_v)$

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
- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007

