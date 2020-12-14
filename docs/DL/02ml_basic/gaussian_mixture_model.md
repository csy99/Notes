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

### 适用场景

多分类。

**优点**

1. 鲁棒性较好

**缺点**

1. 需要数据服从分布（具有严格假设）
2. 参数较多，计算相对比较复杂



# K-平均演算法

我们可以通过K-Means (KMeans)得到每个类的聚类中心$\mu_k$。

### 过程

1. 初始化聚类中心$\mu_k$
2. 将每个数据点分配到离自己最近的聚类中心$z_i = arg min_k ||x_i - \mu_k||_2^2$
3. 更新聚类中心$\mu_k = \frac 1 {N_k} \sum_{i:z_i=k} x_i$

<img src="https://i.postimg.cc/1tW06vs4/kmeans-1.png" height=200>

由于KMeans更擅长发现凸类型簇，聚类结果有可能出现偏差。



# 最大期望算法

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

## 代码实现

使用python最基础的代码。

```python
import numpy as np

# specifying data points
x0 = np.array([[-3], [1]])
x1 = np.array([[-3], [-1]])
x2 = np.array([[3], [-1]])
x3 = np.array([[3], [1]])
data = [x0,x1,x2,x3]
n = len(data)
d = len(x0) ## dimension

# specifying initial condition
K = 2
mu0_start = np.array([[-1], [0]])
mu1_start = np.array([[1], [0]])
mus_start = [mu0_start,mu1_start]
Ident = np.eye(d)
Sigmas_start = [Ident, Ident]
priors_start = [1/K]*K

def calc_responsibility(x_list, prior_list, mu_list, Sigma_list, i, k):
    # used to prevent singular matrix
    rand_mat = 1e-6*np.random.rand(K, K)
    x = x_list[i]
    mu = mu_list[k]
    Sigma = Sigma_list[k]
    prior = prior_list[k]
    r = prior*np.exp(-(x-mu).T@np.linalg.inv(Sigma+rand_mat)@(x-mu)/2)
    total = 0
    for itr in range(len(mu_list)):
        mu_prime = mu_list[itr]
        Sigma_prime = Sigma_list[itr] + rand_mat
        prior_prime = prior_list[itr]
        total += prior_prime*np.exp(-(x-mu_prime).T@np.linalg.inv(Sigma_prime)@(x-mu_prime)/2)
    return (r/total)[0,0]

def calc_mu(x_list, r_list):
    total = np.zeros((d,1))
    for i in range(len(x_list)):
        total += r_list[i]*x_list[i]
    total /= np.sum(r_list)
    return total

def calc_cov_mat(x_list, r_list, mu_):
    size = len(r_list)
    res = np.zeros((K, K))
    for i in range(size):
        tmp = np.outer(x_list[i]-mu_,x_list[i]-mu_)
        res += tmp*r_list[i]
    return res / np.sum(r_list)

def calc_prior(r_list):
    return sum(r_list)/n

def run_EM(data, priors_start, mus_start, Sigmas_start, iters=30, printing=False):
    priors_prev = priors_start
    mus_prev = mus_start
    Sigmas_prev = Sigmas_start
    
    responsibilities = np.zeros((K, n))
    priors = [0]*K
    mus = [0]*K
    Sigmas = [0]*K
    for itr in range(iters):
        # calculate responsbility for each data point for each class
        for k in range(K):
            for i in range(n):
                responsibilities[k][i] = calc_responsibility(data, priors_prev, mus_prev, Sigmas_prev, i, k)
        # calculate class prior
        priors = [calc_prior(responsibilities[k], n) for k in range(K)]
        # calculate class center
        mus = [calc_mu(data, responsibilities[k]) for k in range(K)]
        # calculate class covariance matrix
        Sigmas = [calc_cov_mat(data, responsibilities[k], mus[k]) for k in range(K)]
        
        # update parameters 
        priors_prev = priors
        mus_prev = mus
        Sigmas_prev = Sigmas
        if printing:
            print("-----iteration {}-----".format(itr))
            print("Priors: {}".format(priors))
            print("Mean: ")
            for mu_ in mus:
                print(np.matrix(mu_))
            print("Covariance matrix: ")
            for Sigma_ in Sigmas:
                print(np.matrix(Sigma_))
    return priors, mus, Sigmas
```





# Reference

- Probability and Information Theory in Machine Learning (ECE 601), Matthew Malloy, Fall 2020

