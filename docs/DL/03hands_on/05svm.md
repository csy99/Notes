# Linear SVM Classification

SVM are sensitive to the feature scales. 

### Soft Margin Classification

If we strictly impose that all instances must be on one side, it is called hard margin classification. There are two main issue: 1) only works if the data is linearly separable 2) sensitive to outliers. 

To avoid these issues, use a more flexible model. The objective is to find a good balance between keeping the margin large and limiting the margin violations. It is called soft margin classification.

If our SVM is overfitting, try reducing **C**. 

```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

svm_clf.fit(X, y)
```

SVM classifiers do not output probabilities for each class. 

The LinearSVC class regularizes the bias term, so we should center the training set first by subtracting its mean. For better performance, we need to set **dual** to False, unless there are more features than training instances. 



# Nonlinear SVM Classification 

For some datasets that are not even close to being linearly separable, one approach is to add more features (e.g., polynomial features). 
```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)
```

### Polynomial Kernel

There is a trick in SVM called kernel trick, which makes it possible to get the same result as if we add many poly features. 

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)
```

The hyperparam **coef0** controls how much the model is influenced by high-degree poly versus low-degree poly. 

A common approach to find the right hyperparam values is to use grid search. It is often faster to first do a very coarse grid search, then a finer grid search around the best values found. 

### Similarity Features

Another tech to tackle nonlinear problems is to add features computed using a similarity function, measuring how much each instance resembles a particular landmark. 

Gaussian Radial Basis Function (RBF)

$\phi_\gamma(x, l)=exp(-\gamma||x-l||^2)$

This is a bell-shaped function varying from 0 (far from landmark) to 1. 

The simplest approach to select the landmarks: create a landmark at the location of each and every instance in the dataset. The downside is that a training set of size m\*n will be turned to size m\*m (assuming drop the original features).

### Gaussian RBF Kernel

```python
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)
```

Increasing $\gamma$ makes the curve narrower, each instance's range of influence is smaller: the decision boundary becomes more irregular. If our model is overfitting, reduce it. 

### Computational Complexity

The LinearSVC class is based on liblinear library, which implements an optimized alg for linear SVM. It does not support the kernel trick, but scales well. $O(m*n)$

The SVC class is based on libsvm library, supporting kernel trick.  $O(m^2*n)$ to $O(m^3*n)$. This alg suits complex small or medium-sized training sets. It scales well with the number of features, especially with sparse features. 



# SVM Regression

The objective is reversed: fit as many instances as possible on the street while limiting margin violations. 

```python
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X, y)
```

To tackle nonlinear regression tasks, use kernelized SVM. 

```python
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg.fit(X, y)
```



# Under the Hood

Compute $w^Tx+b$ . The slope is w, the smaller the slope, the larger the margin. 

### Dual

Given a constrained optimization problem, know as the primal problem, it is possible to express a different but closely related problem, called its dual problem. The solution to the dual problem typically gives a lower bound to the solution of the primal problem. Sometimes, it provides the same solution. 

### Kernelized SVM

Suppose we want to apply a second-degree poly transformation to a two-dimensional training set, then train a linear SVM classifier on the transformed data. 

The term of $\phi(a)$ includes $x_1^2, \sqrt{2}x_1x_2, x_2^2$. If we apply this mapping to two 2D vectors, and compute the dot product. We will get  $\phi(a)^T\phi(b) = (a^Tb)^2$

In ML, a kernel is a function capable of computing the dot product $\phi(a)^T\phi(b)$ based only on the original vectors a and b, without having to compute the transformation. 

Mercer's Theorem: if a function K(a, b) respects a few mathematical conditions called Mercer's conditions (e.g., K must be continuous and symmetrical: K(a, b) = K(b, a)) then there exists a function $\phi$ that maps a and b into another space (possibly much higher dim) such that K(a, b) = $\phi(a)^T\phi(b)$. We can use K as a kernel even if we do not know what $\phi$ is. 

### Online SVM

One method is to use GD. But it converges more slowly than the methods based on QP. 

 