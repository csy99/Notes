# Transfer Learning

## Load Data Set

```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
```

```python
(xTrain, xVal), info = tfds.load(
    'cats_vs_dogs', 
    with_info=True, 
    as_supervised=True, 
    split=['train[:80%]', 'train[80%:]'],
)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
```



## Resize Input Images

Different pretrained NNs have different required input image size. 

```python
BATCH_SIZE = 32
dim = 224

def format_image(image, label):
  image = tf.image.resize(image, (dim, dim))/255.0
  return  image, label

num_examples = info.splits['train'].num_examples
train_batches = xTrain.shuffle(buffer_size=num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = xVal.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
```



## Transfer Learning from TensorFlow Hub

```python
url = "https://tfhub.dev/google/tf2-preview/..."
extractor = hub.KerasLayer(url, input_shape=(255, 255, 3))
# disable the training so that all weights kept
extractor.trainable = False
model = tf.keras.Sequential([extractor, layers.Dense(2)])
```



## Save Models

Usually, use timestamp as part of the file name so that it is unique. 

```python
t = time.time()
path = "./model_{}.h5".format(int(t))
model.save(path)
```

Reload the model. 

```python
reloaded = tf.keras.models.load_model({path, custom_objects={'hub.KerasLayer'}})
reloaded.summary()
```



## Export as SavedModel

```python
t = time.time()
path = "./model_{}".format(int(t))
tf.saved_model.save(model, path)
```

Reload a savedmodel. Notice that the object returned by `tf.saved_model.load` is not a Keras object. 

```python
reload_md = tf.saved_model.load(path)
reload_keras = tf.keras.models.load_model(path, custom_objects={'hub.KerasLayer'})
```

Download to local. 

```bash
!zip -r model.zip {path}
```



 