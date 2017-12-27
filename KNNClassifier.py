import matplotlib as mlp
mlp.use('TkAgg')
from matplotlib import cm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from adspy_shared_utilities import plot_fruit_knn #utility python code provided


fruits = pd.read_table('fruit_data_with_colors.txt')

print(fruits.head())

look_up_fruit = dict(zip(fruits.fruit_label, fruits.fruit_name))

print(look_up_fruit)

X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
cmap = cm.get_cmap('gnuplot')

scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)

knn = KNeighborsClassifier(n_neighbors=5)
print(knn.fit(X_train, y_train))
print(knn.score(X_test, y_test))

#ex 1
fruit_prediction_1 = knn.predict([[20, 4.3, 5.5, '0.3']])#predict a class of fruit with weight 20g, width 4.3cm and height 5.5cm and a color score
print(look_up_fruit[fruit_prediction_1[0]])

#ex 2
fruit_prediction_2 = knn.predict([[100, 3.4, 4.5, '0.3']])
print(look_up_fruit[fruit_prediction_2[0]])

plot_fruit_knn(X_train, y_train, 5, 'uniform')
"""
#for different values of k

k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

print(scores)
#print(plt.isinteractive())
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy/score')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()
"""
#for different split proportions
#"""
split = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
knn = KNeighborsClassifier(n_neighbors=5) #here default p = 2 minkowski or euclidean distance in knn
plt.figure()

for s in split:
    #scores = []
    #for i in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
    knn.fit(X_train, y_train)
    scores = knn.score(X_test,y_test)
    plt.plot(s, scores, 'bo')

plt.xlabel('Training % split')
plt.ylabel('Accuracy')
plt.show()

#"""
"""
Issues faced
-->matplotlib not working on mac, so write the line mlp.use('TkAgg') at the top below import
-->pd.scatter is deprecated, so use pd.plotting.scatter
-->pyplot plt doesn't automatically show plot, write the line plt.show()

Key points
--> use shift+tab to show parameters of a function if using jupyter notebook
"""