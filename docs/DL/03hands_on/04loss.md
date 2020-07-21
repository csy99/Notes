# Linear Regression

The normal equation. 

$w = (X^TX)^{-1}X^Ty$

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```

The `LinearRegression` class is based on the `scipy.linalg.lstsq()` function (the name stands for "least squares"). 

```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
```

This function computes $\hat{\theta} = X^+y$, where $X^+$ is the pseudoinverse of X. 

### Computational Complexity

The SVD approach used by LinearRegression class is about $O(n^2)$, where n is the number of features. 



# Gradient Descent

To implement GD, we need to compute the gradient of the cost function with regard to each model parameter $\theta_j$. 

### Stochastic Gradient Descent

Picks a random instance in the training set at every step. 

### Mini-batch GD

Computes  the gradients on small random sets of instances called mini-batches. 



# Polynomial Regression

Add powers of each feature as new features. 

```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```

When there are multiple features, Polynomial Regression is capable of finding relationships between features. 



# Learning Curves

We need to know if our model is overfitting/underfitting. We can plot model's performance on the training set and val set as a function of the training set size or training iteration. 

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)   							       					train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)             
```

We can use a plain LR model. 

```python
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])                         
plt.show()                                      
```

Both curves reached a plateau. They are close and high. Underfit. 

Now we try a 10th-degree polynomial model. 

```python
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])           
plt.show()    
```

The error on the training data is much lower than with the LR model. There is a gap between two curves. Overfit. 



# Regularized Linear Models

### Ridge Regression

The regularization term should only be added to the cost function during training. It is important to scale the data before performing Ridge Reg, as it is sensitive to the scale of the input features. 

We can use a closed-form solution. 

```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

Or, we can use SGD. 

```python
sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```

### Lasso Regression

It tends to eliminate the weights of the least important features. 

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=.1)
lasso_reg.fit(X, y)
```

### Elastic Net

A middle ground between Ridge and Lasso Regression. We can control the mix ration r. When r = 0, Elastic Net is equivalent to Ridge Regression. 

In general, Elastic Net is preferred over Lasso because Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated. 

```python
from sklearn.linear_model import ElasticNet
e_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

### Early Stopping

Stop training as soon as the val error reaches a minimum. 



# Logistic Regression

### Estimate Prob

$\sigma(t) = \frac{1}{1+exp(-t)}$

### Training and cost func

$cost = -log(\hat{p})$ if y =1. $cost = -log(1-\hat{p})$ if y =0. 

The reason why we are using log loss instead of MSE here is that it is a convex function. In addition, it will give larger updates when the error is larger. 

### Decision Boundaries

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

X = iris["data"][:, (2, 3)]  # petal length and width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
save_fig("logistic_regression_contour_plot")
plt.show()
```

LR has been added l2 penalty by default. The hyperparam controlling the regularization strength of LR is not alpha, but its inverse: C. The smaller, the regularization strength is stronger. 



# Softmax Regression

LR can be generalized to support multiple classes directly -> softmax regression. 

Cross entropy gradient vector for class k

$\nabla_{\theta^{(k)}} = \sum_{i=1}^m (\hat{p}_k^{(i)} - y_k^{(i)})x^{(i)}$

```python
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)
```





