A group of predictors is called an enselmble. 

# Voting Classifiers

Aggregate the predictions of each classifier and predict the class that gets the most votes. This majority-vote classifier is called a hard voting classifier. 

Due to the law of large numbers, when we have a sufficient number of diverse weak learners, the ensemble can still be a strong learner. If all classifiers are perfectly independent will be better, not the case here since they are trained on the same data and thus are likely to make the same types of errors. 

```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(xtrain, ytrain)
```

If all classifiers are able to estimate class probabilities, then we can tell scikit-learn to predict the class with the highest class prob, averaged over all individual classifier. This is called soft voting. 

# Bagging and Pasting

We can use same training algorithm for every predictor and train them on different random subsets of the training set. When sampling is performed with replacement, it is called bagging (short for bootstrap aggregating). When w/o replacement, it is called pasting. Generally, after aggregation, the net result has a similar bias but a lower variance. 

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bg_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, 
    max_sampels=100, boot_strp=True, n_jobs=-1
)
bag_clf.fit(xtrain, ytrain)
```

The Bagging Classifier automatically performs soft voting if the base classifier can estimate class probabilities. 

Bagging introduces a bit more diversity in the subsets that each predictor is trained on, so it ends up with a slightly higher bias than pasting and less variance. 

### Out-of-Bag Eval

With Bagging, some instances maybe sampled several times for any given predictor, while others may not be sampled at all. On overage about 63% of the training instances are sampled for each predictor. The remaining 37% are called out-of-bag (oob) instances (not the same 37% for all predictors). As a result, we do not need for a separate validation set. We can set **oob_score=True** to request an automatic oob evaluation after training. 

```python
bag_clf = BaggingClassifier(
	DecisionTreClassifier(), n_estimators=100,
    bootstrap=True, n_jobs_=-1, oob_score=True
)
bag_clf.fit(xtrain, ytrain)
bag_clf.oob_score_
```

# Random Patches and Random Subspaces

Sampling is controlled by two hyperparas: **max_features** and **bootstrap_features**. This tech is useful when we deal with high-dimensional inputs. Sampling both training instances and features is called the Random Patches. Keeping all training instances but sampling features is called the Random Subspaces. 



# RF

RF is an ensemble of Decision Trees, generally trained via the bagging method, with max_samples set to the size of the training set. 

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(xtrain, ytrain)
```

A RF classifier has almost all hyperparameters of a DT classifier plus those of a Bagging classifier. 

The following bagging is similar to RF. 

```python
bag = BaggingClassifier(
	DecisionTreeClassifier(splitter="random", max_leaf_nodes=16), n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
)
```

### extra trees

Extremely Randomized Trees ensemble (extra-trees): We can make trees more random by using random thresholds for each feature rather than searching for the best possible thresholds.

### Feature Importance

We can measure the relative importance of each feature. Scikit-learn measures importance by looking at how much the nodes that use this feature reduce impurity on average. 

```python
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rf_clf.feature_importances_):
    print(name, score)
```



# Boosting

Boosting refers to any Ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting method is to train predictors sequentially, each trying to correct its predecessor. 

### AdaBoost

One way to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. Once all predictors are trained, they have different weights depending on their overall accuracy on the weighted training set. One drawback is they cannot be parallelized. 

Each instance weight is initially set to 1/m. We calculate the weighted error rate of the j-th predictor. 

$r_j=\frac{\sum^m_{i=1, \hat{y}_j^{(i)}\ne y^{(i)}} w^{(i)}}{\sum_{i=1}^m w^{(i)}}$ 

The predictor's weight $\alpha_j$ is then computed using

$\alpha_j = \eta log\frac{1-r_j}{r_j}$

Next, the AdaBoost algorithm updates the instance weights, boosting the weights of the misclassified instances

$w^{(i)} = w^{(i)}exp(\alpha_j)$ if the instance is misclassified. 

Then all the instance weights are normalized. 

Finally, a new predictor is trained using the updated weights, and the whole process is repeated. 

During prediction, it computes the predictions of all the predictors and weighs them using the predictor weights. 

$\hat{y}(x) = argmax_k \sum^N_{j=1, \hat{y}_j(x)=k} \alpha_j$

Scikit-Learn uses a multiclass version of AdaBoost called SAMME (stagewise additive modeling using a multiclass exponential loss function). SAMME.R can estimate class probabilities. 

```python
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
	DecisionTreeClassifier(), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5
)
```

### Gradient Boosting

Instead of tweaking the instance weights at every iteration like AdaBoost, Gradient Boosting tries to fit the new predictor to the residual errors made by the previous predictor. 

Gradient Boosted Regression Trees (GBRT). We can build a simple instance. 

```python
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
y_pred = sum(tree.predict(xtest) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

Instead, we can use scikit-learn implementation. 

```python
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
```

The **learning_rate** scales the contribution of each tree. If we set it to a low value, we need more trees in the ensemble to fit the training set, but the results are usually better (this is a regularization technique called shrinkage). In order to find the optimal number of trees, we can use early stopping. 

```python
xtrain, xval, ytrain, yval = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(xtrain, ytrain)
errors = [mean_squared_error(yval, ypred) for ypred in gbrt.staged_predict(xval)]
bst_n = np.argmin(errors)+1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n)
gbrt_best.fit(xtrain, ytrain)
```

It is also possible to implement early stopping by stopping training early instead of training a large number of trees and then looking back to find the optimal. 

```python
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(xtrain, ytrain)
    y_pred = gbrt.predict(xval)
    val_error = mean_squared_error(yval, ypred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up =0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break
```

The GradientBoostingRegressor class also supports a **subsample** hyperparameter, which specifies the fraction of training instances to be used for training each tree. Stochastic Gradient Boosting. 

### XGBoost

```python
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(xtrain, ytrain, eval_set=[(xval, yval)], 
           early_stopping_rounds=2)
y_pred = xgb_reg.predict(xval)
```



# Stacking

Instead of using trivial functions to aggregate the predictions of all predictors in an ensemble, we can train a model to perform the aggregation. To train the blender, we can use a hold-out set. The first subset of training set is used to train the predictors in the first layer. Next, these predictors are used to make predictions on the second set. We can create a new training set using these predicted values as input features, and keeping the target values. The blender is trained on this new training set. 





