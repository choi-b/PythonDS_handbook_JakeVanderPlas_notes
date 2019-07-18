# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:17:11 2019

@author: 604906
"""

#Data Manipulation Tutorials following
#Jake Vanderplas's PythonDataScienceHandbook
#His Github repo: https://github.com/jakevdp/PythonDataScienceHandbook
#Data found at: https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data


#################################################################
############### Chapter 5, Machine Learning #####################
#################################################################

#Machine Learning: a means of building models of data.
#models with tunable parameters.
#fit -> predict, understand newly observed data.
#1. Supervised - measured features of data & labels to new, unknown data (classification & regression)
#2. Unsupervised - features of data without reference to any label. "Letting the dataset speak for itself" (Clustering & Dimensionality Reduction)
#**. Semi-Supervised - often useful when only incomplete labels are available.

## Introducing Scikit-Learn

#Data as table
#number of rows: n_samples
#number of columns: n_features

#Features matrix : 2-D array/matrix
#shape: [n_samples, n_features]
#often a NumPy array OR a Pandas DataFrame.

#Label/Target Array: "y"
#Usaully 1-dim with length n_samples.
#Continuous numerical values/Discrete classes/labels

#Common steps in using Scikit-Learn
#1. Choose a class of model by importing the estimator.
#2. Choose model hyperparameters
#3. Arrange data into a features matrix and target vector
#4. Fit the model to your data by calling the fit() method
#5. Apply the model to new data. (Predict)


## Supervised Learning Ex: Simple Linear Regression
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.rand(50)
plt.scatter(x, y);

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model

#arrange data into a features matrix and target vector
X = x[:, np.newaxis]
X.shape

#fit your model to your data
model.fit(X, y)
model.coef_
model.intercept_

#Predict labels for unknown data
xfit = np.linspace(-1,11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
#Visualize results by plotting raw data/then model fit
plt.scatter(x, y)
plt.plot(xfit, yfit)


## Supervised Learning Ex: Iris Classification
import seaborn as sns
iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state = 1)
#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain,ytrain)
y_model = model.predict(Xtest)

#use accuracy_score to calculate proportion of correctly predicted labels
from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


## Unsupervised Learning Ex: Iris Dimensionality
#Recall: Iris data is 4-dim

#Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)

#plot the results
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg = False)
#species are fairly well separated, even though
#the PCA algorithm had no knowledge of the species labels.


## Unsupervised Learning Ex: Iris Clustering
#A powerful clustering method: Gaussian Mixture Model (GMM)
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components = 3,
            covariance_type = 'full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)

#Before plotting, add cluster label to the Iris DataFrame
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False)

## Application: Exploring Handwritten Digits
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape
#1797 samples, each with 8x8 grid of pixels.

import matplotlib.pyplot as plt

#visualize the first 100 digits.
fig, axes = plt.subplots(10, 10, figsize=(8,8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')

#We need a 2-dim representation : [n_samples, n_features]
# -> Treat each pixel as a feature!
X = digits.data
X.shape
y = digits.target
y.shape

#Manifold learning algorithm -> transform data into two dimensions
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

#Plot it
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

#Classification on Digits
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

#Use confusion matrix to see which ones we got wrong
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

#lot of confusion over 2 and 8

#Plot the inputs again with labels (green - correct, red - incorrect)
fig, axes = plt.subplots(10, 10, figsize=(8,8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_model[i]) else 'red')
#Get insight into which "shapes" might be confusijng.


## Hyperparameters and Model Validation
# Selecting and Validating your Model

#Holdout sets: hold back some subset of data from the training
#Use this to check model performance

from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                  train_size=0.5)
#fit model on one set
model.fit(X1, y1)

#evaluate model on 2nd set of data
y2_model = model.predict(X2)
accuracy_score(y2, y2_model)

#Cross-Validation: do a sequence of fits where each subset of the data is used 
#both as a training set and as a validation set

#5-fold CV
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)


## Selecting the Best Model

# Bias-Variance Trade-off
#UNDERFIT: High Bias: Performance onvalid. set: SIMILAR to performance on training set.
#OVERFIT: High Variance: Performance on valid. set: FAR WORSE than performance on training set.

#Validation curve
#Training Score & Validation Score

#Ex) Polynomial Regression Model.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
#Create some data
import numpy as np
def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)

#Visualize data along with polynomial fits of several degrees
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1,1.0)
plt.ylim(-2,12)
plt.legend(loc='best')

#Visualize the Validation curve - to see trade-off between bias (underfitting) and variance (overfitting)
#function auto computes both training score and validation score
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree',
                                          degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xlabel('degree')
plt.ylabel('score')

#Compute and Display this fit
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)


## Learning Curves
# - a plot of the training/validation score w.r.t. the size of the training set
#generate new dataset with a factor of five more points
X2, y2 = make_data(200)
plt.scatter(X2.ravel(), y2)

#plot the validation curve for this too
degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2,
                                            'polynomialfeatures__degree',
                                            degree, cv=7)
plt.plot(degree, np.median(train_score2, 1), color='blue',
label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=0.3,
linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=0.3,
linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)

#Dashed line - previous dataset (smaller)
#Solid line - new results

# **Larger dataset can support a much more complicated model.

#<General behavior>
#Model score is always > validation score except by chance
#The curves should keep getting closer together but never cross.

# Learning Curves in Scikit-Learn
from sklearn.model_selection import learning_curve

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),X, y, cv=7,train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray',linestyle='dashed')

    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
    
#LEFT: When the learning curve has already converged, adding more training data will not significantly improve the fit.
#RIGHT: using a more complicated model at the expense of higher model variance. With large enough training data, the learning curve eventually converges.


## Validation in Practice: Grid Search

#Ex) Using Grid Search to find the optimal polynomial model.
from sklearn.model_selection import GridSearchCV

param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)

#Fit to data
grid.fit(X, y)
#Ask for best parameters
grid.best_params_

#Show the fit to our data (same as earlier)
model = grid.best_estimator_

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)


## Feature Engineering ##

# => Taking whatever info you have and turning it into numbers
# to use to build your feature matrix.

#That "whatever info" could be things like
# - categorical data
# - text
# - images

#Discuss "derived features" for increasing model complexity
# & "Imputation" of missing data.
#This process is aka "vectorization" - converting arbitrary data into well-behaved vectors.


#EXAMPLE) Categorical Features
#your data on housing prices could be:
data = [
        {'price': 85000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
        {'price': 70000, 'rooms': 3, 'neighborhood': 'Fremont'},
        {'price': 65000, 'rooms': 3, 'neighborhood': 'Wallingford'},
        {'price': 60000, 'rooms': 2, 'neighborhood': 'Fremont'}
        ]

#Scikit-Learn assumes numerical features => algebraic quantities.

#we must use one-hot encoding.
#since data: list of dictionaries, use DictVectorizer

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)

#See the meaning of each column
vec.get_feature_names()


#EXAMPLE) Text Features
#simplest method: word counts.
sample  = ['problem of evil',
           'evil queen',
           'horizon problem']
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
X

#before inspecting, convert this to a DataFrame with labeled columns.
import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

#Since raw word count -> too much weight on words that appear very frequently.
#TF-IDF (Term frequency-inverse document frequency)
#weights the word counts by a measure of how OFTEN they appear in the documents.

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# Image Features (for more advanced than the digits data we looked at) #
#comprehensive summary -> see Scikit-Image project

# Derived Features #
#Converting a linear reg -> polynom reg by transforming the input (basis function regression)
#good motivational path to KERNEL methods.

#Example)
x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
plt.scatter(x,y)

#Try fitting a reg. line
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X,y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)

#Transform the data: add extra columns of features
#add polynomial features to data
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

#Plot using the input "X2" instead of "X"
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x,y)
plt.plot(x, yfit)

# Imputation of Missing Data #

from numpy import nan
X = np.array([[ nan, 0, 3 ],
              [3, 7, 9],
              [3, 5, 2],
              [4, nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])

#need to replace missing data with some appropriate fill value (imputation)
#Baseline approach: use mean, median, or most frequent value.
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
X2

# Feature Pipelines #

#Pipeline
#1. Impute missing values using the mean
#2. Transform features to quadratic
#3. Fit a linear regression
from sklearn.pipeline import make_pipeline

model = make_pipeline(SimpleImputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())

#the pipeline: standard Scikit-Learn object
#apply to input data
model.fit(X, y)
print(y) 
print(model.predict(X))
#perfect prediction since model was applied to the data it was trained on.