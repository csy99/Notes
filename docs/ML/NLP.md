## Tokenization

```python
from tf.keras.preprocessing.text import Tokenizer
# maximum number of words to keep, based on word frequency. 
# Only the most common `num_words-1` words will be kept.
tok = Tokenizer(num_words=10, oov_token="<OOV>")
tok.fit_on_texts(sentences)
word_idx = tok.word_index # a dictionary
```

**OOV token**

Words that do not appear in dictionary. 



## Text to Sequences

Use padding and truncating to make sequences same length. 

```python
from tf.keras.preprocessing.sequence import pad_sequences
seq = tok.texts_to_sequences(sentences)
# by default, seqs are trucated or padded from the start
padded = pad_sequences(seq, maxlen=10, padding='post', truncating='post')
```



## Word Embeddings

Embeddings are clusters of vectors (represent a given word) in high dimensional space. 

**Benefits**

- easy to compute
- can be visualized

**Drawbacks**

- fail to consider the order

```python
from tf.keras.layers import Embedding
model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length), 
    Flattern(),
    Dense(6)
])
```

In this model, `Flattern()` can be replaced by `GlobalAveragePooling1D()`. Their function is to connect Embedding layer with Dense layer. 



## Subword

**Benefits**

- subwords are more likely to appear in the original dataset

**Drawbacks**

- the meaning may be ambiguous 

```python
import tensorflow_datasets as tfds
vocab_size = 1000
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=5)
```



## RNN

Text can be affected by words both before or after them. 

```python
model = Sequential([
    Embedding(),
    Bidirectional(LSTM(16), return_sequences=True),
    Bidirectional(LSTM(16)),
    Dense()
])
```

**GLUE**

General Language Understanding Evaluation benchmark

a collection of resources for training, evaluating, and analyzing NL understanding systems



## Gated Recurrent Unit (GRU)

has reset gate and update gate

similar to LSTM but does not maintain cell state





## Text Generation

Predict the next word in a sequence. 

- consider memory and output size constraints
- add/subtract from layer sizes or embedding dimensions
- use `np.random.choice` with the prob for more variance in predicted outputs

