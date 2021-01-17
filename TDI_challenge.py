#!/usr/bin/env python
# coding: utf-8

# # TDI challenge
# Jorge Martin del Campo
# 
# 01/17/2021

# ## Setup

# In[1]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# ## Get the Data

# In[7]:


import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[8]:


fetch_housing_data()


# In[9]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[10]:


housing = load_housing_data()
housing.head()


# In[11]:


housing.info()


# In[12]:


housing.ocean_proximity.value_counts()


# In[13]:


housing.describe()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
save_fig("attribute_histogram_plots")
plt.show()


# ## Create a Test Set

# In[16]:


# to make this notebook identical at every run
np.random.seed(42)


# In[17]:


import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[18]:


train_set, test_set = split_train_test(housing, 0.2)
len(train_set)


# In[19]:


len(test_set)


# In[20]:


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[21]:


housing_with_id = housing.reset_index() # adds an 'index' column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')


# In[22]:


housing_with_id['id'] = housing.longitude * 1000 + housing.latitude
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')


# In[23]:


test_set.head()


# In[24]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[25]:


test_set.head()


# In[26]:


housing.median_income.hist()


# In[28]:


housing['income_cat'] = pd.cut(housing.median_income, 
                               bins=[0.,1.5,3.0,4.5,6.,np.inf], 
                               labels=[1,2,3,4,5])


# In[29]:


housing.income_cat.value_counts()


# In[30]:


housing.income_cat.hist()


# Now we are ready to do stratified sampling based on the income category using scikit-learn.

# In[32]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing.income_cat):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[33]:


strat_test_set.income_cat.value_counts() / len(strat_test_set)


# In[34]:


housing.income_cat.value_counts() / len(housing)


# In[35]:


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# In[36]:


compare_props


# In[37]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ## Discover and Visualize the Data to Gain Insights

# In[38]:


# Let's create a copy to explore the data without harming the training set

housing = strat_train_set.copy()


# ### Visualizating Geographical Data

# In[41]:


housing.plot(x='longitude', y='latitude', kind='scatter')
save_fig('bad_visualization_plot')


# In[43]:


housing.plot(x='longitude', y='latitude', kind='scatter', alpha=0.1)
save_fig("better_visualization_plot")


# In[44]:


housing.plot(x='longitude', y='latitude', kind='scatter', alpha=0.4,
            s=housing.population/100, label='population', figsize=(10,7),
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
            sharex=False)
plt.legend()
save_fig('housing_prices_scatterplot')


# In[45]:


# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))


# In[47]:


import matplotlib.image as mpimg

california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()


# ### Looking for Correlations

# In[48]:


# Let's look at the standard correlation coefficient (Pearson's r)

corr_matrix = housing.corr()


# In[49]:


corr_matrix.median_house_value.sort_values(ascending=False)


# In[50]:


from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms',
             'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig('scatter_matrix_plot')


# In[56]:


housing.plot(kind='scatter', x='median_income', y='median_house_value',
            alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig('income_vs_house_value_scatterplot')


# ### Experimentig with Attribute Combinations

# In[57]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[58]:


corr_matrix = housing.corr()
corr_matrix.median_house_value.sort_values(ascending=False)


# In[59]:


housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# In[60]:


housing.describe()


# ## Prepare the Data for Machine Learning Algorithms

# In[62]:


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set.median_house_value.copy()


# ### Data Cleaning

# In[63]:


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[64]:


# Option 1: Get rid of the correspornding districts
sample_incomplete_rows.dropna(subset=['total_bedrooms'])


# In[65]:


# Option 2: Get rid of the whole attribute
sample_incomplete_rows.drop('total_bedrooms', axis=1)


# In[68]:


# Option 3: Set the values to some value (zero, the mean, the median, etc.)
median = housing.total_bedrooms.median()
sample_incomplete_rows.total_bedrooms.fillna(median, inplace=True)
sample_incomplete_rows


# In[69]:


# Scikit-Learn has a handy class to take care of missing values: SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')


# The median ca only be computed on numerical attributes. We will remove the text attribute.

# In[70]:


housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])


# In[71]:


imputer.fit(housing_num)


# In[72]:


imputer.statistics_


# Check that this is the same as manually computing the median of each attribute

# In[73]:


housing_num.median().values


# Transform the training set:

# In[74]:


X = imputer.transform(housing_num)


# In[75]:


# Putting back to data frame format from array.
housing_tr = pd.DataFrame(X, columns=housing_num.columns, 
                          index=housing.index)


# In[76]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[77]:


imputer.strategy


# In[78]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)


# In[79]:


housing_tr.head()


# ### Handling Text and Categorical Attributes
# Now let's preprocess the categorical input feature, ocean_proximity:

# In[80]:


housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)


# In[83]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[84]:


ordinal_encoder.categories_


# In[85]:


# Other way to do it is by using the one-hot encoder method.
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# By default, the OneHotEncoder class returns a sparse array, but we can convert it to a dense array if needed by calling the toarray() method:

# In[86]:


housing_cat_1hot.toarray()


# Alternatively, you can set sparse=False when creating the OneHotEncoder:

# In[87]:


cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[88]:


cat_encoder.categories_


# ### Custom Transformers
# Let's create a custom transformer to add extra attributes:

# In[89]:


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[90]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# ### Transformation Pipelines
# Now let's build a pipeline for preprocessing the numerical attributes:

# In[92]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[93]:


housing_num_tr


# In[94]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)


# In[95]:


housing_prepared


# ## Select and Train a Model

# ### Training and Evaluating on the Training Set

# In[96]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[97]:


# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)

print('Predictions:', lin_reg.predict(some_data_prepared))


# In[98]:


# Compare against the actual values:
print('Labels:', list(some_labels))


# In[99]:


some_data_prepared


# In[100]:


# Let's measure this regression model's RMSE on the whole training set using Scikit-Learn
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[101]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae


# In[102]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[103]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rsme = np.sqrt(tree_mse)
tree_rsme


# ### Better Evaluation Using Cross-Validation

# In[105]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[106]:


# Let's look at the results:
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
    
display_scores(tree_rmse_scores)


# In[107]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[108]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[109]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[110]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[113]:


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# In[114]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# ## Fine-Tune Your Model
# ### Grid Search

# In[115]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[116]:


grid_search.best_params_


# In[117]:


grid_search.best_estimator_


# In[118]:


# Let's look at the score of each hyperparameter combination tested during the grid search:
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# In[119]:


pd.DataFrame(grid_search.cv_results_)


# ### Randomized Search

# In[120]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[121]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ## Analyze the Best Model and Their Errors

# In[122]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[123]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ### Evaluate Your System on the Test Set

# In[124]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[125]:


final_rmse


# We can compute a 95% confidence interval for the test RMSE:

# In[126]:


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# We could compute the interval manually like this:

# In[127]:


m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# Alternatively, we could use a z-scores rather than t-scores:

# In[128]:


zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# ## Extra Material
# ### A full pipeline with both preparation and prediction

# In[129]:


full_pipeline_with_predictor = Pipeline([
    ('Preparation',full_pipeline),
    ('linear',LinearRegression())
])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)


# ### Model persistence using joblib

# In[130]:


my_model = full_pipeline_with_predictor


# In[131]:


import joblib

joblib.dump(my_model, 'my_model.pkl')
my_model_loaded = joblib.load('my_model.pkl')


# ### Example SciPy distributions for RandomizedSearchCV

# In[132]:


from scipy.stats import geom, expon

geom_distrib = geom(0.5).rvs(1000, random_state=42)
expon_distrib = expon(scale=1).rvs(1000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()


# ## Exercise Solutions:
# ### 1.
# Question: Try a Support Vector Machine regressor (sklearn.svm.SVR), with various hyperparameters such as kernel="linear" (with various values for the C hyperparameter) or kernel="rbf" (with various values for the C and gamma hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best SVR predictor perform?

# In[133]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel':['linear'], 'C':[10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
             {'kernel':['rbf'], 'C':[1.0, 3.0, 10., 30., 100., 300., 1000.0],
             'gamma':[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}]
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                          verbose=2)
grid_search.fit(housing_prepared, housing_labels)


# The best model achieves the following score (evaluated using 5-fold cross validation):

# In[134]:


negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse


# That's much worse than the RandomForestRegressor. Let's check the best hyperparameters found:

# In[136]:


grid_search.best_params_


# The linear kernel seems better than the RBF kernel. Notice that the value of C is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for C (removing the smallest values), because it is likely that higher values of C will be better.

# ### 2.
# Question: Try replacing GridSearchCV with RandomizedSearchCV.

# In[137]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

params_distribs = {'kernel':['linear','rbf'],
                   'C':reciprocal(20,200000),
                  'gamma':expon(scale=1.0)}

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, params_distribs, n_iter=50,
                               cv=5, scoring='neg_mean_squared_error',
                               verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# The best model achieves the following score (evaluated using 5-fold cross validation):

# In[138]:


negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse


# Now this is much closer to the performance of the RandomForestRegressor (but not quite there yet). Let's check the best hyperparameters found:

# In[139]:


rnd_search.best_params_


# This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.
# 
# Let's look at the exponential distribution we used, with scale=1.0. Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.

# In[140]:


expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()


# The distribution we used for C looks quite different: the scale of the samples is picked from a uniform distribution within a given range, which is why the right graph, which represents the log of the samples, looks roughly constant. This distribution is useful when you don't have a clue of what the target scale is:

# In[141]:


reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()


# The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be (indeed, as you can see on the figure on the right, all scales are equally likely, within the given range), whereas the exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be.

# ### 3. 
# Question: Try adding a transformer in the preparation pipeline to select only the most important attributes.

# In[142]:


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


# Note: this feature selector assumes that you have already computed the feature importances somehow (for example using a RandomForestRegressor). You may be tempted to compute them directly in the TopFeatureSelector's fit() method, however this would likely slow down grid/randomized search since the feature importances would have to be computed for every hyperparameter combination (unless you implement some sort of cache).

# Let's define the number of top features we want to keep:

# In[143]:


k = 5


# Now let's look for the indices of the top k features:

# In[144]:


top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices


# In[145]:


np.array(attributes)[top_k_feature_indices]


# Let's double check that these are indeed the top k features:

# In[146]:


sorted(zip(feature_importances, attributes), reverse=True)[:k]


# Looking good... Now let's create a new pipeline that runs the previously defined preparation pipeline, and adds top k feature selection:

# In[147]:


preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])


# In[148]:


housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)


# Let's look at the features of the first 3 instances:

# In[149]:


housing_prepared_top_k_features[0:3]


# Now let's double check that these are indeed the top k features:

# In[150]:


housing_prepared[0:3, top_k_feature_indices]


# ### 4.
# Question: Try creating a single pipeline that does the full data preparation plus the final prediction.

# In[151]:


prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])


# In[152]:


prepare_select_and_predict_pipeline.fit(housing, housing_labels)


# In[153]:


some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))


# ### 5.
# Question: Automatically explore some preparation options using GridSearchCV.

# In[154]:


param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)


# In[155]:


grid_search_prep.best_params_


# The best imputer strategy is most_frequent and apparently almost all features are useful (15 out of 16). The last one (ISLAND) seems to just add some noise.

# In[ ]:




