# Vanishing/Exploding Gradients

**vanishing gradients**

Gradients often get smaller and smaller as the algorithm progresses down to the lower layers, and this may leave the lower layers' connection weights unchanged. 

**exploding gradients**

Layers get insanely large weight updates and the algorithm diverges. 

## Initialization

By default, Keras uses Glorot initialization with a uniform distribution. We can change by setting **kernel_initalizer** parameter.

```python
Dense(10, kernel_initializer="he_normal")
```

#### Xavier initialization / Glorot Initialization

The variance of the outputs of each layer to be equal to the variance of its inputs, and need to guarantee equal variance before and after flowing through a layer in the reverse direction. Actually, it is not possible unless the layer has an equal number of inputs and neurons (fan-in = fan-out). A good compromise will be the connection weights of each layer must be initialized randomly where $fan_{avg} = (fan_{in} + fan_{out})/2$.   

前一层节点数越多，设定为目标节点的初始值的权重尺度越小。Xavier初始值是以激活函数是线性函数为前提而推导出来的。因为sigmoid函数和tanh函数左右对称，且中央附近可以视作线性函数，所以适合使用Xavier初始值。

If using logistic activation function:

Normal distribution with mean 0 and variance $\sigma^2 = 1/fan_{avg}$

Uniform distribution between -r and r, with $r = \sqrt{\frac{3}{fan_{avg}}}$. 

#### LeCun initialization

If we replace $fan_{avg}$ with $fan_{in}$ for the formula above, we get LeCun initialization. 

#### He Initialization

The initialization strategy for ReLU activation function and its variants. 

## Nonsaturating Activation Function

#### ReLU Variants

ReLU activation function may suffer from dying ReLUs: keep outputting 0. Instead, we can use leaky ReLU (a small slope for negative values). Setting $\alpha$ to be 0.2 is a good default. Other similar variants include randomized leaky ReLU (where  $\alpha$ is picked randomly), and parametric leaky ReLU ($\alpha$ no longer a hyperparameter, learnt during BP). 

To use leaky ReLU, create a LeakyReLU layer after the layer to be applied

```python
model = Sequential([
    Dense(10, kernel_initializer="he_normal"),
    LeakyReLU(alpha=0.2),
])
```

#### Exponential Linear Unit (ELU)

$ ELU(z) = \alpha(exp(z)-1)$, if z < 0. 

It takes on negative values when z<0, which allows the unit to have an average output closer to 0 and helps alleviate the vanishing gradients problem. If $\alpha$ is 1, then the curve is smooth everywhere. 

#### Scaled ELU (SELU)

If all hidden layers use the SELU activation function, then the NN will self-normalize, which solves the vanishing/exploding gradients. 

Criteria to use

- input must be standardized
- hidden layers' weights init with LeCun normal initialization
- architecture is sequential 
- all layers are dense

To use SELU

```python
model = Sequential([
    Dense(10, activation="selu", kernel_initializer="lecun_normal")
])
```

#### Conclusion

In general, SELU > ELU > leaky ReLU > ReLU > tanh > logistic. Many libraries and hardware accelerators provide ReLU-specific optimizations. Therefore, if speed is priority, choose ReLU. 

## Batch Normalization

The technique adds an operation in the model just before or after the activation function of each hidden layer. It lets the model learn the optimal scale and mean of each of the layer's inputs. 

During test time, we will have no way to compute each input's mean and standard deviation. One solution is to wait until the end of training, then run the whole training set through the NN and compute the mean and std of each input of the BN layer, which can be used during testing. Another solution is to estimate these final statistics by using a moving average. 

In addition, it acts like a regularizer, reducing the need for other regularization techniques. 

Runtime is a little bit slow, but there is some trick to improve the runtime. 

```python
model = Sequential([
    Flatten(input_shape=[28,28]),
    BatchNormalization(),
    Dense(300, activation="elu"),
    BatchNormalization(),
    Dense(10, activation="softmax")
])
```

If adding the BN layers before the activation functions rather than after (shown above), we must remove the activation function from the hidden layers and add them as separate layers after the BN layers. Moreover, since BN layer includes one offset per input, we can remove bias term from previous layer. 

```python
model = Sequential([
    Flatten(input_shape=[28,28]),
    BatchNormalization(),
    Dense(300, kernel_initalizer="he_normal", use_bias=False),
    BatchNormalization(),
    Activation("elu"),
    Dense(300, kernel_initalizer="he_normal", use_bias=False),
    BatchNormalization(),
    Activation("elu"), 
    Dense(10, activation="softmax")
])
```

For hyperparameter, we can tweak momentum. The value should be close to 1. Higher for larger datasets and smaller mini-batches. We can also tweak axis, which determines which axis to be normalized. 

## Gradient Clipping

Clip the gradients during backpropagation so that they never exceed some threshold. Often used in RNN. 

We need to set the **clipvalue** or **clipnorm** argument when creating an optimizer. 

```python
optm = SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optm)
```

Clip by value may change the direction, even though this works well in practice. For example, a gradient vector is [0.9, 100], and we will get [0.9, 1.0] after clipping shown above. If we do not want to change direction, we need to clip by norm. 



# Reusing Pretrained Layers

Transfer Learning: speed up training considerably and require significantly less training data. It works best when the inputs have similar low-level features, especially in deep CNN, which tend to learn feature detectors that are much more general in the lower layers. 

The output layer and some upper hidden layers of the original model are less likely to be as useful as the lower layers. The more training data we have, the more layers we can unfreeze. It is useful to reduce the learning rate when we unfreeze reused layers. 

```python
modelA = keras.models.load_model("modelA.h5")
modelB = Sequential(modelA.layers[:-1])
modelB.add(Dense(1, activation="sigmoid"))
```

In the code shown above, two models share some layers. So when we train one of them, it will also affect the other. To address the issue, clone the model and then copy the weights. 

```
copyA = keras.models.clone_model(modelA)
copyA.set_weights(modelA.get_weights())
```

Since the new output layer was initialized randomly, it will make large errors, and large error gradients may wreck the reused weights. To avoid this, we can freeze the reused layers during the first few epochs. 

```python
for layer in modelB.layers[:-1]
	layer.trainable = False
modelB.compile()
```

We can train the model for a few epochs and then unfreeze the reused layers (which requires compiling the model again).  

### Unsupervised Pretraining

If it is cheap to gather unlabeled training data but expensive to label, we can use them to train an unsupervised model, such as an autoencoder or a generative adversarial network (GAN). 

### Pretrain on an Auxiliary Task

Train a first NN on an auxiliary task for which we can easily obtain training data.



# Faster Optimizers

### Momentum Optimization

It cares about what previous gradients were. The gradient is used for acceleration, not for speed. Momentum is like friction with 0 being high and 1 being low (no friction). Usually, the value is set to 0.9. 

>Alg: 
>
>$m = \beta m - \eta \nabla_\theta J(\theta)$
>
> $\theta = \theta + m$

Gradient descent goes down the steep slope quite fast, but it takes a long time to go through valley. Momentum optimization will roll down the valley faster. 

```python
optm = SGD(lr=0.001, momentm=0.9)
```

### Nesterov Accelerated Gradient

Also called Nesterov momentum optimization. 

>Alg: 
>
>$m = \beta m - \eta \nabla_\theta J(\theta + \beta m)$
>
> $\theta = \theta + m$

The tweak works because in general the momentum vector will point towards the optimum, so it will be slightly more accurate to use the gradient measured a bit farther in that direction. It helps reduce the oscillations and thus NAG converges faster. 

```python
optm = SGD(momentum=0.9, nesterov=True)
```

### AdaGrad

> Alg:
>
> $s = s + \nabla_\theta J(\theta) \odot \nabla_\theta J(\theta)$
>
> $ \theta = \theta - \eta\nabla_\theta J(\theta) \oslash \sqrt{s+\epsilon}$

The first step accumulates the square of the gradients into the vector s. The second step is almost identical to GD, but the gradient vector is scaled down by a factor. This algorithm decays the learning rate, but it does so faster for steep rather than gentler slops. It is an adaptive learning rate.  It often stops too early when training NN. 

### RMSProp

The algorithm fixes the problem of AdaGrad by accumulating only the gradients from the most recent iterations. 

> Alg:
>
> $s = \beta s + (1-\beta)\nabla_\theta J(\theta) \odot \nabla_\theta J(\theta)$
>
> $ \theta = \theta - \eta\nabla_\theta J(\theta) \oslash \sqrt{s+\epsilon}$

The decay rate $\beta$ is typically set to 0.9. 

```python
optm = keras.optimizers.RMSprop(lr=0.01, rho=0.9)
```

### Adam and Nadam

Adam is similar to both momentum optimization and RMSProp. 

```python
optm = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
```

AdaMax replaces the l2 norm with $l_{\infty}$. 

Nadam is Adam optimization plus the Nesterov trick. 

### Learning Rate Sceduling

#### Power scheduling

The learning rate drops at each step, and it reduces the lr more and more slowly. 

```python
optm = SGD(lr=0.01, decay=1e-4)
```

#### Exponential scheduling

Set the lr to $\eta(t)=\eta_00.1^{t/s}$. The lr will gradually drop by a factor of 10 every s steps. 

```python
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0*0.1**(epoch/s)
    return exponential_decay_fn
exp_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = keras.callbacks.LearningRateScheduler(exp_decay_fn)
hist = model.fit(xtrain, ytrain, callbacks=[lr_scheduler])
```

#### Piecewise constant scheduling

Use a constant lr for a number of epochs, then a smaller learning rate for another number of epochs. 

```python
def piecewise_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 20:
        return 0.005
    else:
        return 0.001
```

#### Performance scheduling

Measure the validation error every N steps, and reduce the lr by a factor of $\lambda$ when the error stops dropping.

```python
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
```

#### 1cycle scheduling

 Starts by increasing the initial lr, growing linearly up to maximum learning rate, and then decreases linearly down to initial learning rate, finishing the last few epochs by dropping the rate down by several orders of magnitude. 

# Regularization

### l1 and l2 norm

We often want to apply the same regularizer to all layers. To reduce repeating codes, we can create a thin wrapper for any callable. 

```python
fro functools import partial
RegDense = partial(keras.layers.Dense, 
                  activation="elu", 
                  kernel_regularizer=keras.regularizers.l2(0.01))
model = Sequential([
    Flatten(), 
    RegDense(300), 
    RegDense(100), 
    RegDense(10, activation="softmax")
])
```

### Dropout

At training step, every neuron has a probability p of being temporarily dropped out (may be active during next step). After training, neurons don't get dropped anymore. 

Understand it in this way. A unique NN is generated at each training step, and there are a total of 2^N possible networks. These NN are not independent but different. The resulting NN can be seen as an averaging ensemble of all these smaller NN. In practice, we can apply dropout only to the neurons in the top one to three layers (excluding output layer). 

One important technical detail. We need to multiply each input connection weight by the keep probability (1-p) after training. Alternatively, we can divide each neuron's output by the keep probability during training. 

```python
model = Sequential([
    Flatten(input_shape=[28,28]),
    Dropout(rate=0.2),
    Dense(300), 
    Dropout(rate=0.2),
    Dense(10, activation="softmax")
])
```

Since dropout is only active during training, comparing the training loss and the validation loss can be misleading. Make sure to evaluate the training loss w/o dropout (after training). 

#### Monte Carlo (MC) Dropout

1. connects dropout networks and approximate Bayesian inference, giving solid math justification
2. MC dropout boosts the performance of any trained dropout model w/o having to modify it

```python
y_probas = np.stack[model(xtest, training=True) for sample in range(100)]
y_prob = y_probas.mean(axis=0)
```

Averaging over multiple predictions with dropout on gives us a MC estimate that is generally more reliable. 

If our model contains other layers that behave in a special way during training (e.g.: BatchNormalization), then we should not force training mode like above. Instead, use code below. 

```python
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
```

### Max-Norm Regularization

For each neuron, it constrains the weights of the incoming connections s.t. $||w||_2 \le r$, where r is the max-norm hyperparameter. 

```python
Dense(100, kernel_initializer="he_normal", 
    kernel_contraint=keras.constraints.max_norm(1))
```

