# Clustering

Identify similar instances and assign them to clusters. 

### Application

- customer segmentation

  Cluster customers based on their purchases. Recommender system. 

- data analysis

  Analyze each cluster separately. 

- dimensionality reduction

  See prev article. 

- anomaly detection

  Unusual behavior. For example, fraud detection.  

- semi-supervised learning

  If we only have a few labels, we can perform clustering and propagate the labels to all the instances in the same cluster. 

- search engines

  Search for images that are similar to a reference image. 

- segment an image

  By clustering pixels according to color and replacing each pixel's color with the mean color of its cluster, we can reduce the number of different colors in the image. We can detect the contour of each object. 

### K-Means

Find each blob's center and assign each instance to the closest blob.

```python
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)
kmeans.cluster_centers_
```

Instead of assigning each instance to a single cluster, which is called hard clustering, it can be useful to give each instance a score per cluster (soft clustering). The score can be the distance between the instance and the centroid, or can be a similarity score. 
```python
# measures distance
kmeans.transform(X)
```

If we have a high dimensional dataset, we end up with a k-dim dataset. 

#### Alg

The computational complexity is generally linear with respect to the number of instances m, the number of clusters k, and the number of dimensions n, if data has a clustering structure. Complexity if O(m\*n\*k*l). l is number of iteration. 

#### Centroid Init

If we happen to know approximately where the centroids should be, we can set the **init** hyperparam to a numpy array. 

```python
good_init = np.array([[-3, 3], [3, 2], [1,2]])
kmeans = KMeans(n_clusters=3, init=good_init, n_init=1)
```

Besides, we can run the alg multiple times with different random init and keep the best solution. The number of random init is controlled by **n_init** hyperparam: by default is 10. The metric to measure how good the model is is inertia: mean squared distance between each instance and its closest centroid. 

#### Accelerated K-Means and mini-batch k-Means

It accelerate the alg by avoiding many unnecessary distance calculations. It achieved this by using triangle inequality. This is used by default, unless we set **algorithm** to "full".

Moreover, we can use mini-batches, moving the centroids just slightly  at each iteration. 

```python
from sklearn.cluster import MiniBatchKMeans
minibatch_km = MiniBatchKMeans(n_clusters=4)
minibatch_km = fit(X)
```

#### Find optimal number of clusters

The inertia decreases when k increases, and thus is not a good measurement for choosing good k. A solution is to plot the inertia as a function of the number of clusters k, and choose the elbow point. This tech for choosing best k is rather coarse. 

```python
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow', xy=(4, inertias[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1) )
plt.axis([1, 8.5, 0, 1300])
plt.title("inertia_vs_k_plot")
plt.show()
```

A more precise approach (computationally expensive) is to use the silhouette score, which is the mean silhouette coefficient over all the instances. The coefficient is (b-a)/max(a,b); a is the mean distance to the other instances in the same cluster, and b is the mean nearest cluster distance. The coefficient varies between -1 and +1. Close to +1 means that the instance is well inside its own cluster and far from other clusters, 0 means close to cluster boundary, -1 means may have been assigned wrongly. 

```python
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)

scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
plt.title("silhouette_score_vs_k_plot")
plt.show()
```

Furthermore, we can plot every instances' silhouette coefficient, sorted by the cluster they are assigned to and by the value of the coefficient. This is called a silhouette diagram. The vertical dashed lines represent the silhouette score for each number of clusters. 

```python
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)
    
    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")
    
    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

plt.title("silhouette_analysis_plot")
plt.show()
```

#### Limits of K-Means

We need to run the alg several times to avoid suboptimal, and we need to specify the number of clusters. Moreover, it does not behave well when the clusters have varying sizes, different densities or nonspherical shapes. 

It is important to scale the input features before we run K-Means. It does not guarantee that all the clusters will be nice and spherical, but it generally improves things. 

### Usage for Image Segmentation

Color segmentation. 

```python
from matplotlib.image import imread
image = imread(os.path.join("images", "unsupervised_learning", "ladybug.png"))
image.shape
# there are RGB channels
X = image.reshape(-1, 3)
kmeans = KMeans(n_clsters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```

### Preprocessing

An efficient approach to dim reduction. 

```python
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y=True)
xtrain, xtest, ytrain, ytest = train_test_split(X_digits, y_digits)
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression())
])
pipeline.fit(xtrain, ytrain)
pipeline.score(xtest, ytest)

param_grid = dict(kmeans_n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(xtrain, ytrain)
grid_clf.best_params_
grid_clf.score(xtest, ytest)
```

### Semi-Supervised Learning

We can find the image closest to the centroid (representative images).

```python
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmenas.fit_transform(xtrain)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = xtrain[representative_digit_idx]
```

We can look at each image and manually label it, and then fit these representative images. 

```python
y_representative_digits = np.array([4, 8, 0, 6, ...])
log_reg.fit(X_representative_digits, y_representative_digits)
```

Furthermore, we can use label propagation. 

```python
y_train_propagated = np.empty(len(xtrain), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
    
log_reg.fit(xtrain, ytrain_propagated)
```

The problem is that we propagated each representative instances' label to all the instances in the same cluster, including those located close to the cluster boundaries. We should try only propagate the labels to instances that are closest to the centroids. 

```python
percentile_closest = 20
X_cluster_dist = X_digits_dist[np.arange(len(xtrain)), kmeans.labels_]
for i in range(k):
    in_cluster =(kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_dist = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1
    
partially_propagated = (X_cluster_dist != -1)
Xtrain_partial = xtrain[partially_propagated]
ytrain_partial = ytrain[partally_propagated]
```

#### Active Learning

When a human expert interacts with the learning algorithm, providing labels for specific instances when the alg requests them. 

The model is trained on the labeled instances gathered so far, and makes predictions on all unlabeled instances. Then the instances for which the model is most uncertain are given to the expert to be labeled. Iterate until the performance improvement stops being worth the labeling effort. 

### DBSCAN

Defines clusters as continuous regions of high density. 

#### Alg

For each instance, the alg counts how many instances are located within a small distance $\epsilon$ from it, and the region is called the instance's $\epsilon-neighborhood$. 

If an instance has at least **min_samples** instances in its $\epsilon-neighborhood$, then it is considered a core instance. 

All instances in the neighborhood of a core instance belong to the same cluster. This neighborhood may include other core instances; therefore, a long sequence of neighboring core instances forms a single cluster. 

Other instances are considered an anomaly. 

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=600, noise=0.05)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
# idx of core instance
dbscan.core_sample_indices
# core instances 
dbscan.componenets_
```

However, the DBSCAN class does not have a predict() method, only a fit_predict() method. So we need to choose appropriate classification alg. 

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KneighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
knn.predict(xtest)
knn.predict_proba(xtest)
```

We only trained the classifier on the core instances, but we could also trained it on all instances (includes/excludes the anomalies).

We can introduce a max distance so that instances that are far away from clusters are classified as anomalies. 

```python
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()
```

DBSCAN is robust to outliers and has just two hyperparams. 

However, if the density varies significantly across the clusters, it can be impossible for it to capture all clusters. Its computational complexity is $O(m log m)$.

# Other Clustering Alg

### Agglomerative clustering

At each iteration, it connects the nearest pair of clusters. If we draw a tree with a branch of every pair of clusters that merged, we can get a binary tree of clusters. 

### BIRCH

Balanced Iterative Reducing and Clustering using Hierarchies. Designed specifically for very large datasets. During training, it builds a tree structure containing enough info to quickly assign each new instance to a cluster, w/o storing all the instances in the tree: only use limited memory. 

### Mean-Shift

The alg starts by placing a circle centered on each instance; then for each circle it computes the mean of all instances located within it, and it shifts the circle so that it is centered on the mean. It shifts the circles in the direction of higher density, until each of them find a local density max. Unlike DBSCAN, mean-shift tends to chop clusters into pieces when they have internal density variations. The time complexity is $O(m^2)$.

### Affinity Propagation

Uses a voting system, where instances vote for similar instances to be their representatives. Can detect any number of clusters of different sizes. The time complexity is $O(m^2)$.

### Spectral Clustering

Takes a similarity matrix between the instances and creates a low-dimensional embedding from it. Then, it use another clustering alg in this low-dim space. It can capture complex cluster structures, and it can also be used to cut graphs. It does not scale well to large numbers of instances, and it does not behave well when the clusters have very different size. 



# Gaussian Mixtures

A probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose params are unknow. All instances generated from a single Gaussian dist form a cluster that typically looks like an ellipsoid. 

In the simplest GMM, we must know in advance the number k of Gaussian distributions. For each instance, a cluster is picked randomly from among k clusters. The probability of choosing the j-th cluster is defined by the cluster's weight $\phi^{(j)}$. The index of the cluster chosen for the i-th instance is noted $z^{(i)}$. If this index equals j, meaning assigned to the j-th cluster, the location $x^{(i)}$ of this instance is sampled randomly from Gaussian distribution with mean $\mu^{(j)}$ and covariance matrix $\Sigma^{(j)}$. 

We want to start by estimating the weights $\Phi$ and all the distribution params. 

```python
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
gm.weights_
gm.means_
gm.covariances_
```

This class relies on the Expectation-Maximization alg, which can be treated as an extension of K-Means: not only finds cluster centers, but also their size, shape, and orientation, as well as their relative weights ($\phi$). EM uses soft cluster assignments. 

We can check whether the alg converged and how many iterations it took. 

```python
gm.converged_
gm.n_iter_
```

We can then assign instance to cluster. 

```python
gm.predict(X)
gm.predict_proba(X)
```

GMM is a generative model, so we can sample new instances from it (ordered by cluster index). 

```python
xnew, ynew = gm.sample(10)
```

We can estimate the density of the model at any given location using score_samples(). 

```python
gm.score_samples(X)
```

These are probability densities: take on any positive value. To estimate the prob that an instance will fall within a region, we have to integrate the PDF over that region. 

For higher-dim data, we want to limit the number of params that the alg has to learn. One way is to limit the range of shapes and orientations that the clusters can have. 

Set **covariance_type** hyperparam.

"spherical": all clusters must be spherical, but they can have different diameters. 

"diag": clusters can take on any ellipsoidal shape of any size, but the axes must be parallel to the coordinate axes. 

"tied": all clusters must have the same ellipsoidal shape, size and orientation. 

If the param is "tied" or "full", the time complexity is $O(kmn^2+kn^3)$.

### Anomaly Detection using GM

We need to define what density threshold we want to use. IN manufacturing, the ratio of defective products is usually know. 

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```

A closely related task is novelty detection: it differs from anomaly detection in that the algorithm is assumed to be trained on a clean dataset, uncontaminated by outliers, whereas anomaly detection does not make this assumption. 

### Select the number of clusters

We should find the model that minimizes a theoretical information criterion, such as the Bayesian info criterion or the Akaike info criterion. 

$BIC = log(m)p - 2 log(\hat{L})$

$AIC = 2p - 2 log(\hat{L})$

m is the # instances, p is the # parameters learned by the model, L is the maximized value of the likelihood function of the model. 

Both BIC and AIC penalize models that have more params to learn and reward models that fit the data well. 

```python
gm.bic(X)
gm.aic(X)
```

### Bayesian Gaussian Mixture Models

Gives weights equal (or close) to zero to unnecessary clusters. So, we set the number of clusters to a value greater than the optimal number of clusters. 

```python
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(X)
np.round(bgm.weights_, 2)
```

In this model, the cluster params are treated as latent random variables. So $z$ now includes both the cluster parameters and the cluster assignments. 

The beta distribution is commonly used to model random variables whose values lie within a fixed range. In this case, the range is from 0 to 1. The Stick-Breaking Process is best explained through an example. This process is good when new instances are more likely to join large clusters. 

Prior knowledge about the latent variables $z$ can be encoded in a prob dist $p(z)$ called the prior. We can set the **weight_concentration_prior** param. 

For Bayes' rule, posterior = likelihood*prior/evidence. In GM, evidence p(X) is intractable, since it requires considering all possible combo of cluster param and cluster assignments. 

One solution is variational inference. It picks a family of distributions $q(z;\lambda)$ with its own variational param $\lambda$, then optimize these params to make $q(z)$ a good approx of p(z|X). This is achieved by finding the value of lambda that minimizes the KL divergence from  $q(z)$ to $p(z|X)$.  The KL divergence equation can be written as the log of the evidence minus the evidence lower bound (ELBO). In the end, we want to maximize ELBO. 

# Other Alg for Anomaly and Novelty Detection

### PCA

If we compare the reconstruction error of a normal instance with the reconstruction error of an anomaly, the latter will usually be much larger. 

### Fast  - MCD

It assumes that the normal instances are generated from a single Gaussian distribution (not a mixture). It also assumes that he dataset is contaminated with outliers that were not generated from this Gaussian distribution.

### Isolation Forest

Works well in hi-dim. It builds a RF in which each DT is grown randomly: each node picks a feature randomly, then it picks a random threshold value to split the dataset in two. Anomalies are usually far from other instances, so tend to get isolated in fewer steps than normal instances. 

### Local Outlier Factor

Compares the density of instances around a given instance to the density around its neighbors. An anomaly is often more isolated than its k nearest neighbors.

### One class SVM

Better suit for novelty detection. A kernelized SVM classifier separates two classes by first mapping all the instances to a hi-dim space, then separating the two classes using a linear SVM. Since we have one class only, this alg separates the instances in hi-dim from the origin. In the original space, this corresponds to finding a small region that encompasses all the instances. If a new instance does not fall within the region, it is an anomaly. 



















