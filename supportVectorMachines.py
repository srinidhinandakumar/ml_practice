import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import adspy_shared_utilities as utility
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import (plot_class_regions_for_classifier_subplot)
from adspy_shared_utilities import plot_class_regions_for_classifier

#binary classification
plt.figure()
plt.title('Binary Classification')
X_C1, y_C1 = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, flip_y=0.01, class_sep=0.5, random_state=0)
plt.scatter(X_C1[:, 0], X_C1[:, 1], c=y_C1, marker='o', s=50)  # Number of informative, redundant and repeated features must sum to less than the number of total features
#plt.show()

#Linear Support Vector Machine
#plt.figure()
X_train, X_test, y_train, y_test = train_test_split(X_C1, y_C1, random_state=0)
fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
c=1.0
svc = SVC(C=c, kernel='linear').fit(X_train, y_train)
lsvc = LinearSVC(C=c).fit(X_train, y_train)
title = 'SVC with Linear Kernel and C = {}'.format(c)
utility.plot_class_regions_for_classifier_subplot(svc, X_train, y_train, None, None, title, subaxes)
#plt.show()
#plt.figure()
title = 'Linear SVC with C = {}'.format(c)
utility.plot_class_regions_for_classifier_subplot(lsvc, X_train, y_train, None, None, title, subaxes)
#plt.show()

#Difference between the above two functions
#https://stackoverflow.com/questions/35076586/linearsvc-vs-svckernel-linear-conflicting-arguments

#Multiclass classification
fruits = pd.read_table('fruit_data_with_colors.txt')

feature_names = ['height', 'width', 'mass', 'color_score']
target = ['fruit_label']
target_names_fruits = ['mandarin', 'lemon', 'apple', 'orange']
X = fruits[feature_names]
y = fruits[target]

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state=0)
clf = LinearSVC(C=5, random_state=67).fit(X_train, y_train)
print('Linear SVC coef: ', clf.coef_)
print('Linear SVC intercept: ', clf.intercept_)

"""

#line_colors = cmap(np.linspace(0, 1, 10))
plt.figure(figsize=(6, 6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(colors=['0.8'], name=['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
#cmap_fruits = ListedColormap(colors)

plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
            c=y_fruits_2d, cmap=cmap_fruits, edgecolor='black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b,
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)

plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()
"""

#create colomaps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])


X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers=8, cluster_std = 1.3, random_state=4)
y_D2 = y_D2 % 2
"""
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker='o', s=50, cmap=cmap_bold)
#plt.show()
"""

# Breast cancer dataset for classification
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y=True)

#SVC classification
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
plot_class_regions_for_classifier_subplot(SVC().fit(X_train, y_train), X_train, y_train, None, None, 'Default SVC', subaxes) #by default uses the radial basis function
#plt.show()
plot_class_regions_for_classifier_subplot(SVC(kernel='poly', degree=3).fit(X_train, y_train), X_train, y_train, None, None, 'Degree 3 Poly SVC', subaxes)
#plt.show()

#SVC radial basis function gamma parameter
#higher value of gamma => points need to be closer to be classified into the same class
print('Radial Basis Function SVC with Variable Gamma')
fig, subaxes = plt.subplots(3, 1, figsize=(4,11))
for this_gamma, subplot in (zip([0.1, 1, 10], subaxes)):
    clf = SVC(kernel='rbf', gamma=this_gamma).fit(X_train, y_train)
    title = 'SVC RBF Kernel\nGamma: {:.2f}'.format(this_gamma)
    plot_class_regions_for_classifier(clf, X_train, y_train, None, None, title, subplot)
    plt.tight_layout()
