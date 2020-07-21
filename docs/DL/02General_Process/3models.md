# 常用模型

## 3.1 逻辑回归
广义线性模型。

### 3.1.1 原理

#### 输入

训练集数据$T = {(x_1,y_1) ... (x_N,y_N)}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R^n$

损失函数$Cost(y,f(x))$

正则化参数$\lambda_1$ ，$\lambda_2$

学习速率$\alpha$，$\beta$

#### 输出

逻辑回归模型$\hat f(x)$



#### 损失函数

**单位阶跃函数**

不连续并且不充分光滑

**对数概率函数 Sigmoid**

$P_w(y=j|x) = \frac{exp(x^Tw^{(j)})}{\sum_{k=1}^{K}exp(x^Tw^{(k)})}$



#### 算法3-1 逻辑回归算法

1. 随机初始化$\theta$
2. 计算$\theta_{j+1} = \theta_{j} - \alpha\frac{1}{m}\sum_{i=1}^{m}x_i[h(x_i)-y_i]$

3. 迭代

在迭代求解时使用高效的优化算法，如LBFGS、信赖域算法。这些求解方法是基于批量处理的，无法高效处理超大规模的数据集，也无法对线上模型进行快速实时更新。

随机梯度下降（SGD）是另一种优化方法，比如google的FTRL算法。



#### 算法3-2 FTRL算法

1. 对于$i\in\{i \sim d\}$，初始化$z_i = 0, n_i = 0$

2. 对样本t = 1 to T,

   1. 计算$$x_{t+1,i} = \left\{\begin{aligned} &0 ,&if |z_{t,i}| \leq \lambda_1 \\ & -(\frac{\beta+\sqrt{n_i}}{\alpha}+\lambda_2)^{-1}(z_{t,i} - sign(z_{t,i})\lambda_1),& otherwise\end{aligned}\right.$$ 

   2. 计算$p_t = \sigma(x_tw)$，使用label函数和预测值$p_t$迭代

   3. 对于i

      $g_i = (p_t - y_t)x_i$

      $\sigma_i = \frac{1}{\alpha}(\sqrt{n_i + g_i^2} - \sqrt{n_i})$ 

      $z_i = z_i + g_i - \sigma_iw_{t,i}$

      $n_i = n_i +g_i^2$

3. 迭代

建议$\beta$取1。


