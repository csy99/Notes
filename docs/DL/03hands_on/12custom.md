# Using TF like Numpy

A tensor is very similar to a numpy ndarray: usually a multidimensional array, but can also hold a scalar. 

### Tensors and Operations

We can create a matrix. 

```python
mat = tf.constant([1,2,3], [4,5,6])
mat.shape
mat.dtype
```

Indexing and all sorts of tensor operations work much like in numpy. 

Some functions have a different name. For instance, tf.reduce_mean/sum/max() is equivalent to np.mean(), np.sum(), np.max(). In TF, we must write tf.transpose(mat) instead of mat.T in numpy. There is a reason. In TF, a new tensor is created with its own copy of the transposed data. 

We can apply TF operations to numpy arrays and vice versa. 

```python
a = np.array([2, 4, 5])
tf.constant(a)
tf.square(a)
```

Notice that numpy use 64 bit precision by default, while TF uses 32 bit (which runs faster and uses less RAM). When we create a tensor from a numpy array, set dtype=tf.float32. 

### Type Conversions

TF does not do type conversions automatically. Instead, it just raises an exception. For example, you cannot add a float tensor and an integer tensor, or even add a 32-bit float and a 64-bit float. Use tf.cast() to convert types. 

### Variables

The tf.Tensor values are immutable. So we need tf.Variable to store parameters that may be changed. We can modifiy in place using the assign() method. 

```python
v = tf.Variable([[1, 2, 3], [4, 5, 6]])
v.assign(2*v)
v[0,1].assign(44)
v.scatter_nd_update(indices[[0,0], [1,2]], 
                   updates=[100,200])
```

### Other Data Structures

sparse tensors: effieciently represent tensors containing mostly zeros. 

tensor array: lists of tensors.

ragged tensors: static lists of lists of tensors, where every tensor has the same shape and data type. 

string tensors: byte strings. 

sets: manipulate using tf.sets package.

queues: FIFO queue, PQ, RandomShuffleQueue.



# Customizing Models and Training Alg

Customize a loss function. 

```python
def huber_fn(y_true, y_pred):
    err = y_true - y_pred
    is_small_error = tf.abs(err) < 1
    squared_loss = tf.square(err) / 2
    linear_loss = tf.abs(err) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

```python
model.compile(loss=huber_fn, optimizer="nadam")
```

### saving and loading customized model

Saving a model containing a custom loss function works fine. When we load the model, we need to map the names to the objects. 

```python
model = keras.models.load_model("mymodel.h5", custom_objects={"huber_fn": huber_fn})
```

If we create a function that creates a configured loss function, we have to specify the argument. 

```python
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        err = y_true - y_pred
        is_small_error = tf.abs(err) < threshold
        squared_loss = tf.square(err) / 2
        linear_loss = tf.abs(err)*tf.abs(err) - threshold**2/2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")

model = keras.models.load_model("mymodel.h5", 
     custom_objects={"huber_fn": create_huber(2.0)})
```

We can also create a subclass of keras.losses.Loss. 

```python
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        err = y_true - y_pred
        is_small_err = tf.abs(err) < self.threshold
        squared_loss = tf.square(err) / 2
        linear_loss = self.threshold * tf.abs(err) - self.threshold**2/2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold":self.threshold}
```

The get_config() method returns a dictionary mapping each hyperparameter name to its value. 

We can use any instance of this class when compiling the model.

```python
model.compile(loss=HuberLoss(2.0), optimizer="sgd")

model = keras.models.load_model("mymodel.h5",
             custom_objects={"HuberLoss":HuberLoss})
```

### Custom Other parts

Activation functions, initializers, regularizers, and constraints. 

Losses and metrics are conceptually not the same thing. Losses are used by GD to train a model, so they must be differentiable and their gradients should not be 0 everywhere. In contrast, metrics are used to evaluate a model, so they must be easily interpretable and have 0 gradients everywhere. 

Streaming metric: gradually updated, batch after batch. (e.g.: it manifest overall precision so far instead of current batch).

```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), 
                                     tf.float32))
    def result(self):
        return self.total/self.count
    def get_config(self):
    	base_ = super().get_config()
        return {**base_, "threshold": self.threshold}
```

### Custom Layers

If we want to create a custom layer without any weights. 

```python
exp_layer = keras.layers.Lambda(lambda x: tf.exp(x))
```

If we want to build a custom stateful layer. 

```python
class myDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[batch_input_shape[-1], self.units], initializer="glorot_normal")
        self.bias = self.add_wieght(
        	name="bias", 
            shape=[self.units], 
            initializer="zeros"
        )
        super().build(batch_input_shape)
        
    def call(self, X):
        return self.activation(X@self.kernel + self.bias)
    def compute_out_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])
    def get_config(self):
        base = super().get_config()
        return {**base_config, "units": self.units, 
               "activation": keras.activations.serialize(self.activation)}
```

Unless the layer is dynamic, Keras assumes the output shape is the same as the input shape. 

### Losses and Metrics Based on Model Internals

When we want to define losses based on other parts of our model. For example, reconstruction loss is the mean squared difference between the reconstruction and the inputs. By addition this to the mail loss, we will encourage the model to preserve as much info as possible through the hidden layers. It improves generalization.

```python
class ReconstructReg(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30) for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)
        
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction-inputs))
        self.add_loss(0.55*recon_loss)
        return self.out(Z)
```

The build() method creates an extra dense layer which will be used to reconstruct the inputs. The number of inputs is unknown before the build() is called. 

### Computing Gradients using Autodiff

When we need to find the partial derivative w/o too much trouble. 

```python
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)
gradients = tape.gradient(z, [w1, w2])
```

To save mem, only put the strict minimum inside the tf.GradientTape() block. The tape is auto erased immediately after we call its gradient(). So, we will get an error if we call it more than once. The solution is to make it permanent and erase it by hand. 

```python
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)
del tape
```

By default, the tape only tracks operations involving variables. Use tape.watch() to specify something we want the tape to track.

In some cases, we may want to stop gradients from BP through some part of our NN. 

```python
def f(w1, w2) :
    return 3*w1 + tf.stop_gradient(2*w1*w2)
```

### Custom Training Loops

The fit() method is not flexible enough, e.g.: in wide&deep paper. 

//TODO



# TF functions and graphs

Convert a python func to a TF func. 

```python
def cube(x):
    return x**3

tf_cube = tf.function(cube)
tf_cub(2)
```

Alternatively, we can use tf.function as a decorator. 

```python
@tf.funciton
def tf_cube(x):
    return x**3
```

When we write a custom loss function/metric/layer, Keras automatically converts our function into a TF function. 

By default, a TF function generates a new graph for every unique set of input shapes and data types and caches it for subsequent calls. However, if we pass numerical python values, a new graph will be generated for every distinct value. 

### AutoGraph and Tracing

AutoGraph: Python does not provide any other way to capture control flow statements, so autograph analyzes the functions' code and outputs an upgraded version of that function. After this, control flow statements will be replaced by TF operations (e.g.: tf.while_loop()). 

Next, TD calls this upgraded function, and passes a symbolic tensor. The function will run in graph mode, meaning each TF operation will add a node in the graph to represent itself and its output tensors. 

### TF function rules

If we call any external library, including numpy or even standard library, this call will run only during tracing (will not be a part of the graph. 

If our non-TF code has side effects (such as logging or updating a counter), we should not expect these to occur every time we call the TF function. 

We can wrap Python code in a tf.py_function(), but this will hinder performance. In addition, it reduces portability, as the graph will only run on platforms where Python and right libraries are available . 

If the function creates a stateful TF object (e.g.: variable), it must do so only upon the very first call, or else we will get an exception. If you want to assign a new value to the variable, use assign() rather than "=". Use `for i in tf.range(x)` rather than `for i in range(x)`. 

The source code of our python function should be available to TF.









