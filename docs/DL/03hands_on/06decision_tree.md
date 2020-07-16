# Train and Visualize a Decision Tree

Try to understand how it makes predictions. 

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
iris = load_iris()
X = iris.data[:, 2:] # petal len and wid
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

export_graphviz(tree_clf, out_file=image_path("tree.dot"), feature_names=iris.feature_names[2:], rounded=True, filled=True)
```

Then, we can use the dot command line too from the Graphviz package to convert the file to other formats. 

```shell
dot -Tpng tree.dot -o iris_tree.png
```

Decision Trees are intuitive, thus are called white box models. Random Forests or NN are considered black box models. 

# Making Predictions

Decision Tree models do not require much data preparation (feature scaling or centering).

A node's **samples** attribute counts how many training instances it applies to; **value** attribute tells you how many training instances of each class this node applies to; **gini** attribute measures its impurity: a node is pure if all training instances it applies to belong to the same class. 

Gini impurity $G_i = 1 - \sum_{k=1}^n p_{i,k}^2$

$p_{i,k}$ is the ratio of class k instances among the training instances in the i-th node.  

### Estimating Class Prob

To estimate the probability that an instance belongs to a particular class k, a DT traverses the tree to find the leaf node for this instance, and then it returns the ration of training instances of class k in this node. 

### CART Training Alg

CART: Classification and Regression Tree

Scikit-Learn uses the CART algorithm, which produces only binary trees. The alg works by first splitting the training set into two subsets using a single feature k and a threshold $t_k$. It searches for the pair (k, $t_k$) that produces the purest subsets. 

Cost function for classification

$J(k, t_k) = \frac{1}{m}(m_{left}*G_{left}+m_{right}*G_{right})$

It stops recursing once it reaches the maximum depth or if it cannot find a split that will reduce impurity. This is a greedy algorithm: it does not check whether the split will lead to the lowest possible impurity several levels down. 

### Computational Complexity

Making prediction requires traversing the DT requires roughly $O(log_2(m))$. The training algorithm compares all features on all samples at each other, which results in $O(n*m*log_2(m))$. 

### Gini Impurity or Entropy

By default, the Gini impurity measure is used. We can also use entropy impurity measure by setting the **criterion** to "entropy". 

Entropy

$H_i = -\sum_{k=1, p_{i,k}\ne0}^n p_{i,k} log_2(p_{i,k})$

In most cases, they produce similar trees. When they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees. 

### Regularization Hyperparameter

A parametric model, such as a linear model, has a predetermined number of parameters, so its degree of freedom is limited. DT is a nonparametric model, which is more likely to overfit training data.

Increasing min\_* hyperparams or reduce max\_* hyperparams will regularize the model. Other alg work by first training DT w/o restrictions, then pruning unnecessary nodes. Standard statistical tests such as chi-squared test can be used to estimate the nodes' significance. 

### Regression

```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
```

The CART tries to split the training set in a way that minimizes the MSE. 

### Instability

DT love orthogonal decision boundaries, making them sensitive to training set rotation. One solution is to use PCA, which often results in a better orientation. 

The main issue with DT is that they are sensitive to small variations. Besides, we can get very different models even on the same training data. 







