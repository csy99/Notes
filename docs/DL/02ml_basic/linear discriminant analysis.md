# 线性判别分析

全称是Linear Discriminant Analysis (LDA)。

## 原理

给定训练样例集，通过降维的思路进行分类。将样例投影到一条直线上，使得同类样例的投影点接近，异类样例的投影点尽可能远离。LDA降维最多降到类别数K-1的维数。

#### 输入

训练集数据$D = {(x_1,y_1) ... (x_M,y_M)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R^K$，二分类$y_i \in  \{-1, +1\}$

$X_i, \mu_i, \Sigma_i$ 分别表示第i类样例的集合、均值向量、协方差矩阵。

#### 输出

每个分类的均值向量，各分类数据在总体中所占比例，降维矩阵，降维后各分量的权重。如果是二分类就是{+1, -1}。

#### 损失函数

在下面的属于中b代表between class（类间），w代表within class（类内）。

定义全局散度矩阵$S_t = \sum_{i=1}^M(x_i-\mu)(x_i-\mu)^T$ 

定义类内散度矩阵$S_{w} = \sum_{k=1}^K S_{w_k}$。每个类的散度矩阵$S_{w_k} = \sum_{x \in X_k}(x-\mu_k)(x-\mu_k)^T$ 

定义类间散度矩阵$S_b = S_t - S_w = \sum_{i=1}^K(\mu_i-\mu)(\mu_i-\mu)^T$

最大化目标，两个矩阵的广义瑞丽商$J = \frac {||w^T\mu_0 - w^T\mu_1||_2^2}{w^T\Sigma_0w+w^T\Sigma_1w} = \frac{w^TS_bw}{w^TS_ww}$

定义多分类优化目标 $J(w) = \frac{w^T S_b w}{w^T S_w w}$. 我们希望$argmax_w J(w)$。tr表示矩阵的迹(trace)，是对角线元素总和，我们有时候简化成$J(w) = \frac{tr(w^T S_b w)}{tr(w^T S_w w)}$。

对目标函数进行求导，
$$
\frac{\partial J(w)}{\partial w} = 2 S_b w (w^T S_w w)^{-1} + (w^T S_b w) (w^T S_w w)^{-2}(-2)(S_w w) = 0 \\\\
S_b w (w^T S_w w) - (w^T S_b w) S_w w = 0 \\\\
S_w w = \frac {(w^T S_w w)} {(w^T S_b w)} S_b w \\\\
w = S_w^{-1} S_b w\ (常数被约掉，因为大小不重要)
$$
分子分母只跟w的二次向有关，与w长度无关。如果将问题简化成二分类，则w的方向是

$argmin_w -w^TS_bw$ s. t. $w^TS_ww=1$  => $w = S_w^{-1}(\mu_0-\mu_1)$





# Reference

- 《美团机器学习实践》by美团算法团队，第三章
- 《机器学习》by周志华，第三、四章
- [白板推导系列](https://github.com/shuhuai007/Machine-Learning-Session)，shuhuai007