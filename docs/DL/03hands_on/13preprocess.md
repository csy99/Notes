# Data API

Load data from disk. 

```python
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
```

Just use the below as an example. Easier to understand. 

```python
x = tf.range(10)
data = tf.data.Dataset.from_tensor_slices(x)
```

### Chaining Transformation

Apply transformations.

```python
data = data.repeat(3).batch(7)
for item in data:
    print(item)
```

We first call repeat() method, which returns a new dataset that will repeat the items of the original one three times. 

```python
data = data.map(lambda x: x*2)
data = data.apply(tf.data.experimental.unbatch())
data = data.filter(lambda x: x < 5)
```

Note that the function passed to map() must be convertible to a TF function. The map() method applies a transformation to each item, and the apply() method applies to the whole dataset. 

If we want to look at first few items in the dataset, use take().

```python
for item in data.take(5):
    print(item)
```

### Shuffling

GD works best when the instances in the training set are iid. Using shuffle() to create a new dataset that will start by filling up a buffer with the first items of the source dataset. Specify the buffer size and make sure it is large enough. 

```python
data = data.shuffle(buffer_size=5, seed=1).batch(7)
```

If the dataset is too large, this method may not be efficient. One solution is to shuffle the source data. Or, we can split the source data into multiple files, and then read them in a random order during training. To avoid the cases that instances located in the same file end up close to each other, pick multiple files randomly and read them simultaneously. 

```python
# train_path contains a list of training file paths
file_set = tf.data.Dataset.list_files(tran_filepaths, seed=1)
n_readers = 5
# skip first line which is header row
data = file_set.interleave(lambda path: tf.data.TextLineDataset(path).skip(1),
                          cycle_length=n_readers)
```

For interleaving to work best, it is preferable to have files of identical length; otherwise the ends of the longest files will not be interleaved. 

### Preprocess

```python
n_inputs = 8

def preprocess(line):
    defs = [0]*n_input + [tf.constant([], dtype=tf.float32)] 
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x-x_mean)/x_std, y
```

tf.io.decode_csv() takes two arguments: the line to parse and an array containing the default value for each column in the csv file. For the target column, out code tells TF that this column contains floats, but that there is no default value. It will raise an exception if it encounters a missing value. This function returns a list of scalar tensors (one per column). 

We need to stack all tensors into a 1D array. The y is a 1D tensor array with a single value rather than a scalar tensor. 

```python
def csv_reader_dataset(filepaths, rep=1, n_reader=5, n_read_threads=None, shuffle_buf_size=100, n_parse_threads=5, batch_size=32):
    data = tf.data.Dataset.list_files(filepaths)
    data = data.interleave(lambda path: tf.data.TextLineDataset(path).skip(1), cycle_length=n_readers, num_parallel_calls=n_read_threads)
    data = data.map(preprocess, num_parallel_calls=n_parse_threads)
    data = data.shuffle(shuffle_buf_size).repeat(rep)
    return dataset.batch(batch_size).prefetch(1)
```

### Prefetch

While out training algorithm is working on one batch, the dataset will already be working in parallel on getting the next batch ready. 

If data can fit in memory, we can speed up training by using cache() method. Do this after loading and preprocessing, but before shuffling, repeating, batching, and prefetching. 

### Using dataset

```python
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

model = Sequential()
model.compile()
model.fit(train_set, epochs=10, validation_data=valid_set)
model.evaluate(test_set)
new_set = test_set.take(3).map(lambda x, y: x)
model.predict(new_set)
```

We can also create a TF function performing the whole training loop. 

```python
def train(model, optm, loss_fn, n_epochs):
    train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs)
    for x, y in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(x)
            main_loss = tf.reduce_mean(loss_fn(y, y_pred))
            loss = tf.add_n([main_loss]+model.losses)
            grads = tape.gradient(loss, model.trainable_variables)
            optm.apply_gradients(zip(grads, model.trainable_variables))
```



# TF Record Format

TF's preferred format for storing large amounts of data and reading it efficiently. Binary format. We can create and read a TFRecord file using codes below.

```python
with tf.io.TFRecordWriter("data.tfrecord") as f:
    f.write(b"this is the first")
    f.write(b"this is the second")

filepaths = ["data.tfrecord"]
data = tf.data.TFRecordDataset(filepaths)
for item in data:
    print(item)
```

We can compress them if they are large. 

```python
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("data.tfrecord", options) as f:
    write()

data = tf.data.TFRecordDataset(filepaths, compression_type="GZIP")
```

### protocol buffers

```python
from perso_pb2 import Person
p = Person(name="Al", id=1, email=["a@b.com"])
person.email.append("c@d.com")
s.person.SerializeToString()
p2 = Person()
p2.ParseFromString(s)
p == p2 # return true
```

The main protobuf used in TFRecord is Example protobuf. The definition is below

```python
syntax = "proto3";
message BytesList {repeated bytes value = 1;} 
message FloatList {repeated float value = 1 [packed=true];} 
message Int64List {repeated int64 value = 1 [packed=true];}
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features {map<string, Feature> feature = 1;};
message Example {Features features = 1;};
```

[packed=true] is used for repeated numerical fields, for a more efficient encoding. 

```python
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example
person_example = Example(
features=Features(
	feature={
        "name": Feature(byte_list=BytesList(value=[b"Al"])),
        "id": Feature(int64_list=Int64List(value=[123])),
        "emails": Feature(byte_list=BytesList(value=[b"a@b.com"]))
    }))
```

### load and parse examples

The code below defines a description dictionary, then iterates over the TFRecordDataset and parses the serialized Example protobuf in the dataset. 

```python
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

for serialized_example in tf.data.TFRecordDataset(["contacts.tfrecord"]):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
```

We can parse batch by batch. 

```python
data = tf.data.TFRecordDataset(["c.tfrecord"]).batch(10)
for serialized_examples in data:
    parsed = tf.io.parse_example(serialized_examples, feature_description)
```

### SequenceExample protobuf

Deigned to handle lists of lists. 

Below is the definition. 

```python
message FeatureList {repeated Feature feature = 1;}
message FeatureLists {map<string, FeatureList> feature_list = 1;};
message SequenceExample {
    Features context = 1;
    FeatureLists feature_lists = 2;
};
```

If the feature lists contain seqs of varying sizes, we may want to convert to ragged tensors. 

```python
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
	serialized_sequence_example, 
	context_feature_descriptions, 
	sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
```



# Preprocess the Input Features

Preprocessing includes converting all features into numerical features, generally normalizing them, and more. We can include a preprocessing layer. 

### standardize

```python
means = np.mean(xtrain, axis=0, keepdims=True)
std = np.std(xtrain, axis=0, keepdims=True)
eps = keras.backend.epsilon()
model = Sequential([
    keras.layers.Lambda(lambda inputs: (inputs-means)/(std+eps)),
    [...]
])
```

Or, we can use a nice self-contained custom layer. 

```python
class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.mean_ = np.mean(data_sample, axis=0, keepdims=True)
        self.std_ = np.std(data_sample, axis=0, keepdims=True)
    def call(self, inputs):
        return (inputs-self.mean_)/(self.std_+keras.backend.epsilon())
```

Before we use the layer, we need to adapt it to our dataset. 

```python
std_layer = Standardization()
std_layer.adapt(data_sample)
```

The data sample must be large enough to be representative of our dataset, but does not need to be he full set. 

### one hot encoding

```python
vocab = ["c1", "c2", "c3"]
indices = tf.range(len(vocab), dtype=tf.int32)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
```

The reason we use oov buckets: the number of categories is large, or the dataset keeps changing. One solution is to define the vocab based on a data sample (not whole training set) and add some oov buckets for other categories that were not in the sample. 

We can look up in the table to encode in one-hot vectors. 

```python
categories = tf.constant(["c3", "c4", "c1"])
cat_indices = table.lookup(categories)
cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab)+num_oov_buckets)
```

### embedding

An embedding is a trainable dense vector that represents a category. Training tends to make embeddings useful representations of the categories (representation learning).

```python
embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov], embedding_dim)
embedding_matrix = tf.Variable(embed_init)
tf.nn.embedding_lookup(embedding_matrix, cat_indices)
```

Embedding layer can handle the embedding matrix. When the layer is created, it initializes the embedding matrix randomly, and then when it is called with some category indices, it returns the rows at those indices in the matrix. 

```python
embedding = keras.layers.Embedding(input_dim=len(vocab) + num_oov, output_dim = embedding_dim)
embedding(Cat_indices)
```

```python
regular_inputs = Input(shape=[9])
categories = Input(shape=[], dtype=tf.string)
cat_indices = Lambda(lambda c: table.lookup(c))(categories)
cat_embed = Embedding(input_dim=6, output_dim=2)(cat_indices)
encoded_inputs = keras.layers.concatenate([reular_inputs, cat_embed])
outputs = Dense(1)(encoded_inputs)
model = Model(inputs=[regular_inputs, categories],
             outputs=[outputs])
```

One hot encoding followed by a Dense layer without activation function and biases is equivalent to an Embedding layer. However, the latter uses fewer computations. 

### Keras Prep Layers

Normalization: standardization

TextVectorization: encoding. Also have an option to output word-count vectors instead of word indices. 

Discretization: chop continuous data into different bins and encode each bin as a one-hot vector. 



# TF Transform

If preprocessing is computationally expensive, we can handle it before training. In this way, the data will be preprocessed just once per instance before training, rather than once per instance and per epoch during training. 

Consider the case where we wanna deploy the model to mobile and web browsers. Use TensorFlow Extended (TFX). 

```python
import tensorflow_transform as tft

def preprocess(inputs):
    median_age = inputs["housing_median_age"]
    ocean_proximity = inputs["ocean_proximity"]
    standardized_age = tft.scale_to_z_score(median_age)
    ocean_proximity_id = tft.compute_and_apply_vocabulary(ocean_proximity)
    return {
        "standadized_median_age": standardized_age, 
        "ocean_proximity_id": ocean_proximity_id
    }
```

We can apply this function to the whole training set using Apache Beam. 



# TensorFlow Datasets Project

TFDS is not bundled with TF. 

Every item in the dataset is a dictionary containing both the features and labels. 

```python
import tensorflow_datasets as tfds
dataset = tfds.load(name="mnist")
train, test = dataset["train"], dataset["test"]
train = train.shuffle(100).batch(32)
train = train.map(lambda item: (item["image"], item["label"]))
train = train.prefetch(1)
```

We can also ask the load function to do this for us. 

```python
dataset = tfds.load(name="mnist", batch_size=32, 
                    as_supervised=True)
train = dataset["train"].prefetch(1)      
```

