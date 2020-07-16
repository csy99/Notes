In some cases, reducing the dimensionality of the training data may filter out some noise and unnecessary details and thus result in higher performance, but in general it only speeds up training. 

# Curse of Dimensionality 

Most points in a high-dimensional hypercube are very close to the border. High-dimensional datasets are at risk of being very sparse. (If we picked randomly in a 1,000,000 dimensional hypercube, the average dist is about 408). The more dimensions the training set has, the greater the risk of overfitting it. 

# Main Approaches for Dim Reduction

### Projection

Usually, training instances are not spread out uniformly across all dimensions. 

In some cases, the subspace may twist and turn (e.g.: Swiss roll), and we should unroll it to obtain the 2D dataset. 

### Manifold Learning

Work by modeling the manifold on which the training instances lie. It relies on the manifold assumption that most real world high-dim datasets lie close to a much lower-dimensional manifold. 



# PCA

First, it identifies the hyperplane that lies closest to the data, and then it projects the data onto it. 

### Preserving the Var

Select the axis that preserves the maximum amount of variance (lose less information than other projects). Another way to think about it is the axis that minimizes the mean square distance between the original dataset and its projection onto that axis. 

### Principal Components

The i-th axis is called the i-th principal component of the data. We can use SVD to decompose the training set matrix X into the matrix multiplication of three matrics U, $\Sigma$, $V^T$. V contains the unit vectors that define all PC. 

```python
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
```

PCA assumes that dataset is centered around the origin. Scikit-learn's PCA classes take care of this. 

### Project

To project the training set onto the hyperplane and obtain a reduced dataset of dimensionality d, compute multiplication of the original matrix by the new matrix defined by first d columns of V. 

```python
W2 = Vt.T[:, :2]
X2d = X_centered@W2
```

### Using scikit learn

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2d = pca.fit_transform(X)
```

### Explained Variance Ratio

The ration indicates the proportion of the dataset's variance that lies along each principal component. 

```python
pca.eplained_variance_ratio_
```

### Choosing the right dim

The following code performs PCA without reducing dimensionality, then computes the minimum number of dimensions required to preserve 95% of the var. We can then run PCA again. 

```python
pca = PCA()
pca.fit(xtrain)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum>=0.95) + 1
```

A better option. 

```python
pca = PCA(n_components=0.95)
x_reduced = pca.fit_transform(xtrain)
```

Another option is to plot the explained variance as a function of the number of dimensions. There will usually be an elbow in the curve. 

### Compression

It is also possible to decompress the reduced dataset back to high dim by applying the inverse transformation of the PCA projection (not exactly the original data, since some info is lost). The mean squared dist btw the original data and reconstructed data is called the reconstruction error. 

```python
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(xtrain)
X_recovered = pca.inverse_transform(X_reduced)
```

### Randomized PCA

scikit learn  can use a stochastic algorithm called Randomized PCA that quickly finds an approximation of the first d principal components. Its computational complexity is $O(m*d^2) + O(d^3)$ rather than  $O(m*n^2) + O(n^3)$. 

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized")
```

By default, the mode is "auto". If m or n is greater than 500, and d is less than 80% of m or n, then randomized PCA will be used. 

### Incremental PCA

The preceding implementations of PCA requires the whole training set to fir in memory in order for the alg to run. Using IPCA, we can split the training set into mini-batches. This is useful for large training sets and for applying PCA online. 

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=100)
for x_batch in np.array_split(xtrain, n_batches):
    inc_pca.partial_fit(x_batch)
X_reduced = inc_pca.transform(xtrain)
```

Alternatively, we can use memmap class to manipulate a large array stored in a binary file on disk as if it were in memory. 

```python
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```



# Kernel PCA

Kernel trick: match tech that implicitly maps instances into a very high-dim space, enabling nonlinear classification and reg with SVM. A linear decision boundary in the high-dimensional feature space corresponds to a complex nonlinear decision boundary in the original space. 

Kernel PCA can preserve clusters of instances after projection, or sometimes even unrolling datasets that lie close to a twisted manifold. 

```python
from sklearn decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```

### Tuning Hyperparam

kPCA is an unsupervised learning alg, so there is no obvious performance measure to select the best kernel and hyperparams. Dimensionality reduction is often a preparation step for a supervised learning task, so we can use grid search to select the kernel and hyperparameters that lead to the best performance on that task. 

```python
from sklearn.model_seleciton import GridSearchCV
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("kcpa", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])
param_grid = [{
    "kpca_gamma": np.linspace(0.03, 0.05, 10),
    "kpca_kernel":["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(grid_search.best_params)
```

Another approach, entirely unsupervised: select the combo that yields the lowest reconstruction error. Note that reconstruction is not as easy as with linear PCA. We can ask scikit learn to take care of this for us. 

```python
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
mean_squared_error(X, X_preimage)
```



# Locally Linear Embedding (LLE)

Another powerful nonlinear dimensionality reduction technique. It is a manifold learning technique that does not rely on projections. It first measures how each training instance linearly relates to its closest neighbors, and then looking for a low dim representation of the training set where these local relationships are best preserved. This makes it particularly good at unrolling twisted manifolds. 

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```

Scikit-learn's LLE implementation has the following computational complexity: $O(m log(m)n log(k))$ for finding the k nearest neighbors, $O(mnk^3)$ for optimizing the weights, and $O(dm^2)$ for constructing the low-dimensional representations. The last term makes this algorithm scale poorly. 

# Others

### Random Projections

The quality of the dimensionality reduction depends on the number of instances and the target dimensionality, not on the initial dimensionality. 

### Multidimensional Scaling

Reduces dimensionality while trying to preserve the distances between the instances. 

### Isomap

Creates a graph by connecting each instance to its nearest neighbors, then reduces dim while trying to preserve the geodesic dist between the instances. 

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

Keep similar instances close and dissimilar instances apart. Mostly used for visualization. 

### Linear Discriminant Analysis (LDA)

Learns the most discriminative axes between the classes, and these axes can then be used to define a hyperplane onto which to project the data. The projection will keep classes as far apart as possible. 













 