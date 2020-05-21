## Forecast

### Fixed Partitioning

Split the whole dataset into training, validation, and test period in time sequence. 

### Roll-Forward Partitioning

Only use a small subset as training set and move forward every week or 10 days to mimic the real life process. 



## Time Windows

```python
## drop_remainder get rid of last few windows that contains less elements
data = tf.data.DataSet.from_tensor_slices()
data = data.window(5, shift=1, drop_remainder=True)
data = data.flat_map(lambda win: win.batch(5))
for win in data:
    print(val.numpy())
    
## use first few as training data and last one as test data
data = data.map(lambda win: (win[:-1], win[-1:]))
data = data.shuffle(buffer_size=10)
## prefetch allows later elements to be prepared while the current one is being processed
data = data.batch(2).prefetch(1) 
for x, y in data:
    print(x.numpy(), y.numpy())
```



## RNN

Tuning learning rate is tricky for RNN. If it is too high, the RNN will stop learning; if it is too low, the RNN will converge very slowly. 

```python
lr_schedule = keras.callbacks.LearningRateScheduler(lambda ep: 1e-7 * 10 ** (e/20))
model.compile()
hist = model.fit()
plt.semilogx(hist.history["lr"], hist.history["loss"])
```



The loss is going up and downs during training, very unpredictable. Not a good idea to use a small number for early stop. 

```python
es = keras.callbacks.EarlyStopping(patience=50)
checkpoint = keras.callbacks.ModelCheckpoint("md.h5", save_best_only=True)
model.fit(train_set, epochs=500, callbacks=[es, checkpoint])
```



### Stateless RNN

At each training iteration, it starts at a zero state and will drop its state after making prediction. 



### Stateful RNN

The first window is placed at the beginning of the series. The final state vector is preserved for the next training batch, which is located immediately after the previous one. 

**Benefits**

- learn long term patterns

**Drawbacks**

- data set is prepared differently
- training can be slow
- consecutive training batches are very correlated, BP may not work well

```python
def seq_window(series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size+1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size+1))
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(1).prefetch(1) ## use batch=1
```

```python
model = keras.models.Sequential([
    keras.layers.SimpleRNN(100, return_sequences=True, stateful=True, batch_input_shape=[1,None,1]),
    keras.layers.SimpleRNN(100, return_sequences=True, stateful=True),
    keras.layers.Dense(1)
])
```

We need manually set the state to zero state at the beginning of each epoch. 

```python
class ResetState(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()
reset_ = ResetState()
model.fit(callbacks=[es, checkpoint, reset_])
```



## LSTM

Forget Gate: learn when to forget/preserve

Input Gate: output 1, output  0 

Output Gate: 

![LSTM Cell](.\pics\lstm.png)

```python
model = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, 
                     stateful=True, batch_input_shape=[1,None,1])
    keras.layers.LSTM(100, return_sequences=True, stateful=True),
    keras.layers.Dense(1)
])
```



## CNN

We can also use 1D Conv Net in time series prediction. 

```python
model = keras.models.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=5,
                       strides=1, padding="causal",
                       activation="relu",
                       input_shape=[None,1]),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.Dense(1)
])
```

Small dilation let layers learn short term patterns, while large dilation ley layers learn long term patterns. 

```python
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None,1]))
for dilation in [1,2,4,8,16]:
    model.add(
    	keras.layers.Conv1D(dilation_rate=dilation)
    )
model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
```

