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



