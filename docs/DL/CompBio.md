# References

- https://towardsdatascience.com/tagged/stats-ml-life-sciences

# Clustering

## Genre

### HAC

Sensitive to noise. 

### Centroid-based clustering

K-menas, Gaussian Mixture models. 

Only handle clusters with spherical or ellipsoidal symmetry. 

### Graph-based clustering

Spectral, SNN-cliq, Seurat

Robust for high dimensional data. They use the **distance on a graph**, e.g. the number of shared neighbors, which is more meaningful in high dimensions compared to the Euclidean distance. However, to build the graph this method still uses the Euclidean distance. 

### Density-based clustering

Mean-Shift, DBSCAN, OPTICS, HDBSCAN

They allow clustering without specifying the number of clusters. Cluster shape independent and capture any topology of scRNAseq data. 

## Tuning

### HDBSCAN

Only one hyperparameter **minPts** which is the minimal number of points in a cluster. Tuning by minimizing the amount of unassigned cells. It is relatively fast for large data sets, detects outlying cells, and for each cell it reports a probability of assignment to a cluster. 

# Normalize Single Cell

The spurious correlation of ratios the genes are not supposed to be correlated but they are after the Library Size Normalization.

The problem with Library Size Normalization is that the counts across genes for a given sample sum up to 1, i.e. they are not independent any more but constrained, so they become compositional. By constructing compositional data you are no longer in the Euclidean space which traditional Frequentist statistics is based on, since the Euclidean space / distance is the consequence of the Gaussian distribution. In fact, you end up in the Simplex Space where traditional statistical methods are not applicable as the distance between two points is not Euclidean any more but “Euclidean with a constraint” which is the Aitchison distance.

If you still want to use e.g. PCA on compositional data, you should perform one of the log-ratio transformations for converting the data from the Simplex back to the Euclidean space: additive (alr), center (clr) or isomeric (ilr) log-ratio transforms.

### TPM Normalization

By definition, TPM counts for a given sample sum up to one million, that is a Simplex Space constraint. So TPM counts are not better than Library Size normalized counts as they also suffer from the Simplex Space bias, and one should not naively apply e.g. PCA on those counts. 

### Single Cell

A peculiarity of scRNAseq data is that they contain in contrast to bulk RNAseq large amounts of **stochastic zeros** owing to low amounts of RNA per cell and imperfect RNA capture efficiency (**dropout**). Therefore the TMM and DESeq **size factors** may become unreliably **inflated or equal to zero**. Therefore new single cell specific normalization methods were developed accounting for the large amounts of zeros in the scRNAseq data.



# Select Features for OMICs

### Integrative OMICs

Next Generation Sequencing (NGS) technologies. One challenge is that we assume that the OMICs data should have synergistic effects which allows to more accurately model the behavior of biological cells. Another challenge is that combining different types of biological information increases the number of analyzed features while keeping the number of statistical observations (samples) constant. 

### Univariate Feature Selection

Problems:

1. Univariate feature selection does not defeat the Curse of Dimensionality issue since FDR correction is not enough for this purpose, i.e. it is prone to overfitting and has a poor generalization.
2. Univariate feature selection does not account for multicollinearity between features, i.e. when features are strongly correlated with each other.

### Multivariate Feature Selection

The simplest way to simultaneously account for all explanatory variables in the X matrix would be to put them together into the multiple or multivariate linear model and perform the Ordinary Least Squares (OLS) minimization.

LASSO is that it can not fully handle multi-collinearity among predictors. If two variables are strongly correlated, LASSO will select only one of them (by chance) and set the coefficient in front of the other one to zero. Sometimes this type of feature selection can be problematic if it happens that the feature that was ignored / omitted has more physical / biological meaning than the one which was selected by LASSO. This problem can be avoided with Ridge penalty, in addition Ridge is much more stable for numerical minimization as it provides a fully convex manifold in the high-dimensional space. However, in ultra high-dimensional spaces Ridge can be too permissive and select many noisy features which might not be desirable. Elastic Net penalty provides a compromise between LASSO and Ridge and is generally preferred and recommended by Machine Learning practitioners. 

Another elegant multivariate feature selection technique is the Partial Least Squares (PLS) regression and discriminant analysis which is also called (by its author) Projection on Latent Structures (PLS).



# Reduce Dimension for Single Cell

Single cell genomics is a high dimensional data with approximately 20 000 dimensions corresponding to the protein coding genes. Usually not all of the genes are equally important for the cellular function, i.e. there are redundant genes which can be eliminated from the analysis in order to simplify the data complexity. 

Single cell data has highly non-linear structure, so PCA does not work well. Typically, single cell data have 60–80% zero elements in the expression matrix. In this way, single cell data are similar to image data where e.g. for the images of hand-written digits MNIST data set we have 86% of pixels having zero intensity.

### Linear Discriminant Analysis (LDA)

A popular linear dimensionality reduction techniques. The linear dimension reduction techniques are good at preserving the global structure of the data (connections between all the data points) while it seems that for single cell data it is more important to keep the local structure of the data.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage='auto', solver='eigen', store_covariance=False, tol=1e-4)
lda = model.fit_transform(X, Y)
plt.scatter(lda[:, 0], lda[:, 1], c=Y, cmap='viridis', s=1)
plt.xlabel("LDA1", fontsize = 20); plt.ylabel("LDA2", fontsize = 20)
feature_importances = pd.DataFrame({'Gene':np.array(expr.columns)[:-1], 
                                    'Score':abs(model.coef_[0])})
print(feature_importances.sort_values('Score', ascending = False).head(20))
```

### tSNE

Non-linear dimensionality reduction technique. 

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
X_reduced = PCA(n_components = 30).fit_transform(X)
model = TSNE(learning_rate = 10, n_components = 2, random_state = 123, perplexity = 30)
tsne = model.fit_transform(X_reduced)
plt.scatter(tsne[:, 0], tsne[:, 1], c = Y, cmap = 'viridis', s = 1)
plt.title('tSNE on PCA', fontsize = 20)
plt.xlabel("tSNE1", fontsize = 20)
plt.ylabel("tSNE2", fontsize = 20)
```

#### Shortcomings

1. does not scale well for rapidly increasing sample sizes in scRNAseq
2. does not preserve global data structure
3. only for visualization purposes
4. performs a non-parametric mapping from high to low dimensions, meaning that it does not leverage features (aka PCA loadings) that drive the observed clustering
5. can not work with high-dimensional data directly, Autoencoder or PCA are often used for performing a pre-dimensionality reduction before plugging it into the tSNE
6. consumes too much memory

#### Tuning Param

**Number of PCs**

Reduce initial number of dimensions linearly down to 30-50 latent variables (**initial_dims**) and use this as a new data set for feeding into tSNE. 

We can calculate the p-value of how the observed variance is different from the permuted variance for each PC. 

**Optimal Perplexity**

Typical values for the perplexity range between 5 and 50.  A larger / denser data set requires a larger perplexity. However, too large perplexities will lead to one big clump of points without any clustering. Perplexity ~ N^(1/2). 

**Number of Iterations**

The optimal number of iterations should provide the largest distance between the data points of ~100 units. Until the plot won't change much. 

### UMAP

#### difference compared to tSNE

1. UMAP uses exponential probability distribution in high dimensions but not necessarily Euclidean distances like tSNE but rather any distance can be plugged in. 
2. In addition, the probabilities are not normalized. This saves time for high dim. 
3. UMAP uses the number of nearest neighbors instead of perplexity.
4. UMAP uses a slightly different symmetrization of the high-dimensional probability
5. UMAP uses binary cross-entropy (CE) as a cost function instead of the KL-divergence like tSNE does. It is capable of capturing the global data structure in contrast to tSNE that can only model the local structure at moderate perplexity values.
6. UMAP uses the Stochastic Gradient Descent (SGD) instead of the regular Gradient Descent (GD) like tSNE / FItSNE, this both speeds up the computations and consumes less memory.