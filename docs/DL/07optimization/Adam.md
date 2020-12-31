# Adam算法

### 提出动机

Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均。

### 算法

给定超参数$0 \le \beta_1 < 1$（算法作者建议设为0.9），和超参数$0 \le \beta_2 < 1$（算法作者建议设为0.999）。

对每次迭代做如下改动
$$
v_t = \beta_1 v_{t-1} + (1-\beta_1) g_t \\\\
s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t \circ g_t \\\\
$$
由于我们将$v$和$s$都初始化成0。在时间步$t$我们得到
$$
v_t = (1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i
$$
将过去各时间步小批量随机梯度的权值相加，可以得到
$$
(1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} = 1 - \beta_1^t
$$
需要注意的是，如果$t$较小，过去各时间步小批量随机梯度权值之和会较小。例如，当$\beta_1=0.9$时，$v_1=0.1g_1$。为了消除这样的影响，对于任意时间步$t$，我们可以将$v_t$再除以$(1-\beta_1)$，从而使过去各时间步小批量随机梯度权值之和为1。这也叫作偏差修正。在Adam算法中，我们做偏差修正如下
$$
\hat v_t = \frac{v_t}{1 - \beta_1^t} \\\\
\hat s_t = \frac{s_t}{1 - \beta_2^t} \\\\
$$
在每一步对学习率进行调整。
$$
g_t' = \frac{\eta \hat v_t}{\sqrt{\hat s_t} + \epsilon} \\\\
x_t = x_{t-1} - g_t'
$$

### 代码实现

```python
def init_rmsprop_states(dim=2):
    v_w = np.zeros((dim, 1))
    v_b = np.zeros(1)
    s_w = np.zeros((dim, 1))
    s_b = np.zeros(1)
    return (v_w, s_w), (v_b, s_b)

def rmsprop(params, states, hyperparams, eps=1e-8):
    beta1 = hyperparams['beta1']
    beta2 = hyperparams['beta2']
    for p, (v,s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad * p.grad
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```



# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第7章