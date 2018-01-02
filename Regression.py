import numpy as np
import pandas as pd
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from adspy_shared_utilities import load_crime_dataset #utility python code provided
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)

fruits = pd.read_table('fruit_data_with_colors.txt')

feature_names = ['height', 'width', 'mass', 'color_score']
target = ['fruit_label']

X = fruits[feature_names]
y = fruits[target]

target_fruits = ['mandarin', 'lemon', 'apple', 'orange']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#simple regression with one feature
plt.figure()
plt.title('Sample regression with one feature')
X_R1, y_R1 = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, bias=150.0, noise=30, random_state=0)
plt.scatter(X_R1, y_R1, marker='o', s=50)
#plt.show()

#complex regression
plt.figure()
plt.title('Sample complex regression')
X_F1, y_F1 = make_friedman1(n_samples=100, n_features=10, noise=30, random_state=0)  #n_features must atleast be 5
plt.scatter(X_F1[:, 2], y_F1, marker='o', s=50)  # X and y have to be the same size, so X_F1[:, 2]
#plt.show()

#binary classification
plt.figure()
plt.title('Binary Classification')
X_C1, y_C1 = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, flip_y=0.01, class_sep=0.5, random_state=0)
plt.scatter(X_C1[:, 0], X_C1[:, 1], c=y_C1, marker='o', s=50)  # Number of informative, redundant and repeated features must sum to less than the number of total features
#plt.show()

#Linear Regression

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print('linear model coefficient (w): {}'.format(lr.coef_))
print('linear model intercept (b): {:.3f}'.format(lr.intercept_))
print('R-squared score (training): {:.3f}'.format(lr.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(lr.score(X_test, y_test)))

#Linear Regression Example
plt.figure(figsize=(5, 4))
plt.scatter(X_R1, y_R1, marker='o', s=50, alpha=0.8)
plt.plot(X_R1, lr.coef_*X_R1 + lr.intercept_, 'r-')
plt.title('Least Squares Linear Regressions')
plt.xlabel('Features x')
plt.ylabel('Target y')
#plt.show()

(X_crime, y_crime) = load_crime_dataset()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state=0)
print('Linear  Regression')

lr = LinearRegression().fit(X_train, y_train)
print('Linear model coefficient: {}', format(lr.coef_))
print('Linear model intercept: {:.3f}', format(lr.intercept_))
print('Training score: {:.3f}', format(lr.score(X_train, y_train)))
print('Test score: {:.3f}', format(lr.score(X_test, y_test)))

#Ridge Regression
print('Linear Ridge Regression')
linridge = Ridge(alpha=20.0).fit(X_train,y_train)
print('Linear ridge model coefficient: {}', format(linridge.coef_))
print('Linear ridge model intercept: {:.3f}', format(linridge.intercept_))
print('Training score: {:.3f}', format(linridge.score(X_train, y_train)))
print('Test score: {:.3f}', format(linridge.score(X_test, y_test)))
print('Number of non zero features : ', format(np.sum(linridge.coef_ != 0)))

#MinMax Feature Normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
# X_train_scaled = scaler.fit_transform(X_train) #more efficient way to fit and transform
X_test_scaled = scaler.transform(X_test)
linearScaled = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
linearScore = linearScaled.score(X_test_scaled, y_test)
print('\nFeature Normalized Score: {:.3f}', format(linearScore))

#Alpha affects score
linearScaled = Ridge(alpha=0.0).fit(X_train_scaled, y_train)
linearScore = linearScaled.score(X_test_scaled, y_test)
print('Feature Normalized Score for alpha = 0.0 : {:.3f}', format(linearScore))
linearScaled = Ridge(alpha=10.0).fit(X_train_scaled, y_train)
linearScore = linearScaled.score(X_test_scaled, y_test)
print('Feature Normalized Score for alpha = 10.0 : {:.3f}', format(linearScore))
linearScaled = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
linearScore = linearScaled.score(X_test_scaled, y_test)
print('Feature Normalized Score for alpha = 20.0 : {:.3f}', format(linearScore))
linearScaled = Ridge(alpha=30.0).fit(X_train_scaled, y_train)
linearScore = linearScaled.score(X_test_scaled, y_test)
print('Feature Normalized Score for alpha = 30.0 : {:.3f}', format(linearScore))
linearScaled = Ridge(alpha=40.0).fit(X_train_scaled, y_train)
linearScore = linearScaled.score(X_test_scaled, y_test)
print('Feature Normalized Score for alpha = 40.0 : {:.3f}', format(linearScore))

#Lasso Regression
linlasso = Lasso(alpha=2.0, max_iter=10000).fit(X_train_scaled, y_train)
print('Coef (w): ', format(linlasso.coef_))
print('Training score: ', format(linlasso.score(X_train_scaled, y_train)))
print('Test score: ', format(linlasso.score(X_test_scaled, y_test)))
print('Intercept (b): ', format(linlasso.intercept_))
print('Number of non zero features: ', format(np.sum(linlasso.coef_ != 0)))

#Polynomial Regression
X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1, random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))

poly = PolynomialFeatures(degree=2)# upto quadratic degree
X_poly_F1 = poly.fit_transform(X_F1)
X_train, X_test, y_train, y_test = train_test_split(X_poly_F1, y_F1, random_state=0)
polyreglin = LinearRegression().fit(X_train, y_train)
polyregrid = Ridge().fit(X_train, y_train)

print('Linear Polynomial Regression ')
print('Coef: '.format(polyreglin.coef_))
print('Intercept: ', polyreglin.intercept_)
print('Train score: ', polyreglin.score(X_train, y_train))
print('Test score: ', polyreglin.score(X_test, y_test))

print('Ridge Polynomial Regression ')
print('Coef: ', polyregrid.coef_)
print('Intercept: ', polyregrid.intercept_)
print('Train score: ', polyregrid.score(X_train, y_train))
print('Test score: ', polyregrid.score(X_test, y_test))

#logistic regression for fruits data set
fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
y_apple = y_fruits_2d == 1 #choosing all apples

X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d.as_matrix(), y_apple.as_matrix(), random_state=0)
logistic = LogisticRegression(C=100).fit(X_train, y_train)
plot_class_regions_for_classifier_subplot(logistic, X_train, y_train, None, None, 'Logistic Regression for Apples v/s Other Fruits', subaxes)
print('Fruit with height 6 width 8 is classified as ', ['apple', 'not an apple'][logistic.predict([[6, 8]])[0]])
print('Accuracy on training data {} \nAccuracy on test data {}'.format(logistic.score(X_train,y_train), logistic.score(X_test, y_test)))

#logistic regression on classification simple data set
X_train, X_test, y_train, y_test = train_test_split(X_C1, y_C1, random_state=0)
logistic_simple = LogisticRegression().fit(X_train, y_train)
plot_class_regions_for_classifier_subplot(logistic, X_train, y_train, None, None, 'Logistic Regression for Simple Data Set', subaxes)
print('Accuracy for training {}\nAccuracy for test {}'.format(logistic_simple.score(X_train, y_train), logistic_simple.score(X_test, y_test)))

plt.show()