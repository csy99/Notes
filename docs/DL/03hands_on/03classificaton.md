# MNIST

We can fetch the dataset. 

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X, y = mnist.data, mnist.target
# ML alg expects numbers
y = y.astype(np.uint8)
xtrain, xtest, ytrain, ytest = x[:60000], X[60000:], y[:60000], y[60000:]
```



# Binary Classifier

Simplify the problem to identify only one digit. 

```python
ytrain_5 = ytrain == 5
ytest_5 = ytest == 5
```

We choose SGD to train it (working well on large dataset).

```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=4)
sgd_clf.fit(xtrain, ytrain_5)
sgd_clf.predict([xtest])
```



# Performance Measures

### Cross Validation

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, xtrain, ytrain_5, cv=3, scoring="accuracy")
```

### Confusion Matrix

First, we need a set of predictions so that they can be compared to the actual targets. 

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
ytrain_pred = cross_val_predict(sgd_clf, xtrain, ytrain_5, cv=3)
confusion_matrix(ytrain_5, ytrain_pred)
```

Each row in a confusion matrix represents an actual class, while each column represents a predicted class. The order is TN, FP, FN, TP. 

精确率 Precision = $\frac{TP}{TP+FP}$
召回率 Recall = $\frac{TP}{TP+FN}$  

```python
from sklearn.metrics import precision_score, recall_score
precision_score(ytrain_5, ytrain_pred)
recall_score(ytrain_5, ytrain_pred)
```

Sometimes, we combine the two and get F1 score. 

$\frac{2}{F_1} = \frac{1}{P} + \frac{1}{R}$

```python
from sklearn.metrics import f1_score
f1_score(ytrain_5, ytrain_pred)
```

The F1 score favors classifiers that have similar precision and recall - not always what we want. There is precision/recall trade-off. 

### ROC Curve

ROC curve plots the true positive rate against false positive rate. 

```python
from sklearn.metrics import roc_curve
y_scores = sgd_clf.decision_function([some_digit])
fpr, tpr, thresholds = roc_curve(ytrain_5, y_scores) 

def plot_roc(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plot.plot([0,1], [0,1], 'k--')
```

The dotted line represents the ROC curve of a purely random classifier, and a good classifier stays as far away from that line as possible. 

We can compute AUC. 

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(ytrain_5, y_scores)
```

We should prefer the PR curve whenever the positive class is rare or when we care more about the false positives than the false negatives. Otherwise, use ROC. 



# Multiclass Classification

 One-versus-the-rest (OvR): get the decision score from each classifier for that image and select the class whose classifier outputs the highest score

One-versus-one (OvO): train a binary classifier for every pair of digits. The main advantage of OvO is that each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish. 

We can force scikit-learn to use one of these two alg. 

```python
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
```



# Error Analysis

We can look at confusion matrix in plots. 

```python
y_train_pred = cross_val_predict(sgd_clf, xtrain_scaled, ytrain, cv=3)
conf_max = confusion_matrix(ytrain, ytrain_pred)
plt.matshow(conf_max, cmap=plt.cm.gray)
```

We need to divide each val in the confusion matrix by the number of images in the corresponding class. 

```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
```

The matrix is not necessarily symmetrical. 

Analyzing individual errors can gain insights, but more difficult and time-consuming. 

```python
cl_a, cl_b = 3, 5
X_aa = xtrain[(ytrain==cl_a)&(ytrain_pred==cl_a)]
X_ab = xtrain[(ytrain==cl_a)&(ytrain_pred==cl_b)]
X_ba = xtrain[(ytrain==cl_b)&(ytrain_pred==cl_a)]
X_bb = xtrain[(ytrain==cl_b)&(ytrain_pred==cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
save_fig("error_analysis_digits_plot")
plt.show()
```

The main diff between 3s and 5s is the position of the small line that joins the top line to the bottom arc. We can reduce the 3/5 confusion by centering the images. 



# Multilabel Classification

Output multiple classes for each instance. 

```python
ytrain_large = (ytrain > 6)
ytrain_odd = (ytrain%2==1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(xtrain, y_multilabel)
```

We can measure the F1 score for each ind label then compute the avg. 

```python
ytrain_knn_pred = cross_val_predict(knn_clf, xtrain, y_multilabel, cv=3)
f1_score(y_multilabel, ytrain_knn_pred, average="macro")
```

If we want to assign more weights to one of the class, change **average** to "weighted". 



# Multioutput Classification

Each label can be multiclass. 

We can build a system that removes noise from images. The classifier's output is multilabel (one label per pixel) and each label can have multiple values (pixel intensity ranges from 0-255). Can also be treated as a regression problem. 

```python
noise = np.random.randint(0, 100, (len(xtrain), 784))
xtrain_mod = xtrain + noise
noise = np.random.randint(0, 100, (len(xtest), 784))
xtest_mod = X_test + noise
ytrain_mod = xtrain
ytest_mod = xtest
```

We train the classifier and make the pic clean. 

```python
knn_clf.fit(xtrain_mod, ytrain_mod)
clean_dig = knn_clf.predict([xtest_mod[indices]])
plot_digit(clean_dig)
```



