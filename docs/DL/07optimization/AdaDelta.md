# AdaDelta算法

### 提出动机

除了RMSProp算法以外，AdaDelta算法的提出也是为了解决AdaGrad算法在迭代后期较难找到有用解的问题。

### 算法

AdaDelta算法也像RMSProp算法一样，使用了小批量随机梯度$g_t$按元素平方的指数加权移动平均变量$s_t$，这里的$\rho$类似RMSProp中的$\gamma$。但有意思的是，AdaDelta算法没有学习率这一超参数。另外，AdaDelta算法还维护一个额外的状态变量$\Delta x_t$，其元素同样在时间步0时被初始化为0。

对每次迭代做如下改动
$$
s_t = \rho s_{t-1} + (1-\rho) g_t \circ g_t \\\\
g_t' = \sqrt{\frac{\Delta x_{t-1}+\epsilon}{s_t+\epsilon}} \circ g_t\\\\
x_t = x_{t-1} - g_t' \\\\
\Delta x_{t} = \rho \Delta x_{t-1} + (1-\rho) g_t' \circ g_t'
$$
可以看到，如不考虑$\epsilon$的影响，AdaDelta算法与RMSProp算法的不同之处在于使用$\sqrt{\Delta x_{t−1}}$来替代超参数$\eta$。$\rho$的取值一般在$[0.9,0.99]$。

### 代码实现

```python
def init_adadelta_states(dim=2):
    s_w = np.zeros((dim, 1))
    s_b = np.zeros(1)
    delta_w = np.zeros((dim, 1))
    delta_b = np.zeros(1)
    return (s_w, delta_w), (s_b, delta_b)

def adagrad(params, states, hyperparams, eps=1e-5):
    rho = hyperparams['rho']
    for p, (s,delta) in zip(params, states):
        s[:] += rho * s + (1 - rho) * p.grad * p.grad
        g = (math.sqrt(delta + eps) / (math.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```



# Reference

- [Dive Into Deep Learning](http://zh.gluon.ai/chapter_recurrent-neural-networks/rnn.html)，第7章