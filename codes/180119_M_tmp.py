# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:33:37 2019

@author: Matias Ijäs
"""

"""
Model

Create images of each parameter by classes (10)
Find the most important parameter for each class by comparing to other classes
Give weights to each parameter with given class
Count probability of test sample by choosing maximum of "probability*weight"

Test with X_train samples as well

"""

# %% Importing

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# %% Reading data

x_train = np.load("x_train_kaggle.npy")
y_train_data = np.genfromtxt("groups.csv", delimiter = ',', dtype=[('id',np.uint), ('group_id',np.uint), ('surface','S22')])
y_train = y_train_data['surface']
xx_test = np.load("x_test_kaggle.npy")

# %% Transform data

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

splitter = model_selection.GroupShuffleSplit(n_splits=36, test_size=0.2)
tmp_sp = splitter.get_n_splits(X=x_train, y=y_train, groups=y_train_data['group_id'])
# tmp [X_train, X_test, Y_train, Y_test]

for train_i, test_i in tmp_sp.split(x_train, y_train, [y_train_data['group_id']]):
    print("TRAIN:", train_i, "TEST:", test_i)
    aX_train, aX_test = x_train[train_i], x_train[test_i]
    ay_train, ay_test = y_train[train_i], y_train[test_i]
    print(aX_train, aX_test, ay_train, ay_test)

# %% 

X_train = np.array([x.ravel() for x in X_train])
X_test = np.array([x.ravel() for x in X_test])

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
y_predlda = [0] * len(Y_test)

for x in range(len(X_test)):
    y_predlda[x] = lda.predict([X_test[x,:]])
    
print("lda: ", accuracy_score(Y_test, y_predlda))

# %% Dividing y_train and x_train to classes
y_classes = {}
x_classes = {}

for i in y_train:
    item = i[1]
    if item in y_classes:
        y_classes[item].append(i[0])
    else:
        y_classes[item] = [i[0]]
    if (item,0) in x_classes:
        for j in range(10):
            x_classes[(item,j)].append([x_train[i[0],j,:]])
    else:
        for j in range(10):
            x_classes[(item,j)] = [x_train[i[0],j,:]]
        
# %% Plot x_train to classes

avgs = {}
maxs = {}
mins = {}
drawmarks = ['bo', 'bx', 'go', 'gx', 'ro', 'rx', 'ko', 'kx', 'yo', 'yx']
index = 0
tmp_ind = 0.0
    
for i in x_classes:
    
    a_sum = 0
    a_total = 0
    a_max = None
    a_min = None
    
    for j in x_classes[(i)]:
        val = np.mean(j)
        a_sum += val
        a_total += 1
        if a_max == None or a_max < val:
            a_max = val
        if a_min == None or a_min > val:
            a_min = val
    # print(i, a_sum, a_total, a_max, a_min, a_sum/a_total, "\n")
    avgs[i] = a_sum/a_total
    maxs[i] = a_max
    mins[i] = a_min
    
    """
    # Individual images per material
    plt.plot(a_min, index, drawmarks[index])
    plt.plot(a_max, index, drawmarks[index])
    plt.plot(a_sum/a_total, index, drawmarks[index])
    index += 1
    if index == 10:
        plt.show()
        index = 0
        print(i)
        
    """
    # Total image
    #plt.plot(a_min, index + tmp_ind, drawmarks[index])
    #plt.plot(a_max, index + tmp_ind, drawmarks[index])
    plt.plot(a_sum/a_total, index + tmp_ind, drawmarks[index])
    index += 1
    if index == 10:
        index = 0
        tmp_ind += 0.1
        print(i)
plt.show()
        
# %% Next step
        
