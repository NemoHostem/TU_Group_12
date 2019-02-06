# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:38:03 2019

@author: jonik
"""

import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

path = '../data/'

# 1)
test = np.load(path + 'X_test_kaggle.npy')
train_data = np.load(path + 'X_train_kaggle.npy')
train_labels = np.loadtxt(path + 'y_train_final_kaggle.csv', delimiter=',',dtype=np.str)
groups = np.loadtxt(path + 'groups.csv', delimiter=',',dtype=np.str)
groups = groups[:,1]

# 2)

le = preprocessing.LabelEncoder()
le.fit(train_labels[:,1])
train_labels = le.fit_transform(train_labels[:,1])

# 3)

train, test = next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=0).split(train_data, groups=groups))
X_train, X_test, y_train, y_test = train_data[train], train_data[test], train_labels[train], train_labels[test]

# 4)
"""
X_train = np.ravel(X_train)
X_train = X_train.reshape((1356,1280))
X_test = np.ravel(X_test)
X_test = X_test.reshape((347,1280))


X_train = np.mean(tr_data, axis=2)
X_test = np.mean(te_data, axis=2)
"""
X_train = np.concatenate((np.mean(X_train,axis=2), np.std(X_train,axis=2)), axis=1)
X_test = np.concatenate((np.mean(X_test,axis=2), np.std(X_test,axis=2)), axis=1)


# 5) 
classifiers = [[LinearDiscriminantAnalysis(), 'LDA'],
                [SVC(kernel='linear'),'SVC'],
                [LogisticRegression(solver='lbfgs', multi_class='multinomial',max_iter=2000), 'LogisticRegression'],
                [RandomForestClassifier(n_estimators=900), 'RandomForest']]

for clf, name in classifiers:
    clf.fit(X_train, y_train)
    x_pred = clf.predict(X_test)
    print("Accuracy with", name, accuracy_score(x_pred, y_test))
    