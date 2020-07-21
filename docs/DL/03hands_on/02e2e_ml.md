A piece of info fed to a ML system is called a **signal**. We want a high signal-to-noise ratio. 

A sequence of data processing components is called a data **pipeline**. 



# Get the data

### Create a Test set 

Sometimes, we want to reduce a sampling bias produced by random sampling methods. We should try stratified sampling: the pplt is divided into homogeneous subgroups called strata, and the right number of instances are sampled from each stratum to guarantee that the test set is representative of the overall pplt. 

```python
from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
# Now remove the attribute so data is back to original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```



# Discover and Visualize to Gain Insights

### Visualize Geographical Data

```python
housing.plot(
    kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
```

### Looking for Correlations

If the dataset is not large, we can compute the standard correlation coefficient between every pair of attributes. The standard correlation only measures linear correlations. 

```python
corr_mat = housing.corr()
corr_mat["house_value"].sort_values(ascending=False)
```

Another way to check for correlation between attributes is to use pandas. 

```python
from pd.plotting import scatter_matrix
attr = ["house_value", "income", "total_room"]
scatter_matrix(housing[attr], figsize=())
```

We should choose the most promising attribute to take a closer look. 

```python
housing.plot(kind="scatter", x="income", y="house_value", alpha=0.1)
```

If we find a few data quirks, we may want to clean them up. 

### Attribute Combinations

Similar to feature enginneering. 





# Prepare the Data for ML

### Data Cleaning

```python
housing.dropna(["total_bedrooms"], axis=1)
median = housing["total_bedrooms"].median()
housing.total_bedrooms.fillna(media, inplace=True)
```

Scikit-learn provides a handy class to take care of missing values. The median can only be computed on numerical attributes, so we need to create a copy of the data w/o the text attr. SimpleImputer simply computed the median of each attribute and stored the result in its **statistics_** variable. 

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
# transform the training set by replacing missing val
X = imputer.transform(housing_num)
housing_df = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

### Handling Text and Categorical Attributes

First look at categorical attr. 

```python
housing_cat = housing[["proximity"]]
housing_cat.head()
```

If we find that there are a limited number of possible values, convert from text to number. One solution is OrdinalEncoder. 

```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_enc = OrdinalEncoder()
housing_cat_encoded = ordinal_enc.fit_transform(housing_cat)
ordinal_enc.categories_
```

One issue with this representation is that ML will assume that two nearby values are more similar than two distant values. This may be fine in some cases (e.g.: for ordered categories such as bad-average-good-excellent). 

One solution is One Hot encoder. 

```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat = housing_cat_1hot.toarray()
housing_cat_1hot.categories_
```

If we have a huge number of possible categories, then one-hot encoding will create too many input features. We can replace the categorical input with useful numerical features. We can replace each category with embedding. 

### Custom Transformer

We can create a class and implement fit(), transform(), and fit_transform().

```python
from sklearn.base import BaseEstimator, TansformerMixin

rooms_idx, bedroom_idx, pplt_idx, household_idx = 3,4,5,6

class CombinedAttrAdder(BaseEstimator, TansformerMixin):
    def __init__(self, add_bedroom_per_room=True):
        self.add_bedroom_per_room = add_bedroom_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_idx]/X[:, household_idx]
        pplt_per_household = X[:, pplt_idx]/X[:, household_idx]
        if self.add_bedroom_per_room:
            bedroom_per_room = X[:, bedroom_idx]/X[:, rooms_idx]
            return np.c_[X, rooms_per_household, pplt_per_household, bedroom_per_room]
        else:
            return np.c_[X, rooms_per_household, pplt_per_household]
    
attr_adder = CombinedAttrAdder(False)
housing_extra_attr = attr_adder.transform(housing.values)
```

### Feature Scaling

#### Min-max scaling (normalization)

values are shifted and rescaled so that they end up ranging from 0 to 1. Use MinMaxScaler. 

NN often expect an input value ranging from 0 to 1. 

#### Standardization

subtracts the mean value and then divides by the std. Standardization is less affected by outliers. Use StandardScaler. 

### Transformation Pipelines

Data transformation steps executed in the right order. 

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ("attribs_adder", combinedAttrAdder()), 
    ("std_scaer", StandadScaler())
])
housing_df = num_pipeline.fit_transform(housing)
```

All but the last estimator must be transformers (they must have a fit_transform() method). 

It would be convenient to have a single transformer to handle all columns (both categorical and numerical). Use ColumnTransformer. 

```python
from sklearn.compose import ColumnTransformer
num_attr = list(housing_num)
cat_attr = list(housing_cat)

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attr), 
    ("cat", OneHotEncoder(), cat_attribs)
])
housing_prepared = full_pipeline.fit_transform(housing)
```



# Select and Train a model

### Training Set

```python
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_pred = lin_reg.predict(housing_prepared)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_score = np.sqrt(-scores)
```

We should save every model so that we can come back later to find them. 

```python
import joblib
joblib.dump(my_model, "model.pkl")
# later
my_model = joblib.load("model.pkl")
```

### Fine Tune

#### Grid Search

We can try out consecutive powers of 10. 

```python
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators':[3, 10, 30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimatrors':[3,10], 'max_features':[2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

This tells the alg to first evaluate all 3\*4 combo. Then, with the **bootstrap** = False, try out another 2\*3 combo. 

```python
grid_search.best_estimator_
cvres = grid_search.cv_results_
for score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

#### Randomized Search

When hyperparam search space is large, use Randomized search instead. 

#### Ensemble

Try to combine the models that perform best. 

### Analyze the Best Models

```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attr = ["rooms_per_hold", ..]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attr = list(cat_enc.categories_[0])
attributes = num_attr + extra_attr + cat_one_hot_attr
sorted(zip(feature_importances, attributes), reverse=True)
```

### Evaluate on Test Set

```python
final_model = grid_search.best_estimator_
xtest_prepared = full_pipeline.transform(xtest)
final_pred = final_model.predict(xtest_prepared)
final_mse = mean_squared_error(ytest, final_pred)
```

We can also compute a 95% CI for the generalization error using scipy.stats.t.interval(). 

```python
from scipy import stats
confidence = 0.95
sqr_err = (final_pred - ytest) ** 2
np.sqrt(scipy.stats.t.interval(confidence, len(sqr_err)-1, loc=sqr_err.mean(), scale=scipy.stats.sem(sqr_err)))
```

