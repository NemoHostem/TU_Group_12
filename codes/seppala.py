# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:05:45 2019

@author: jonis
"""

#%%

import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import discriminant_analysis
from sklearn.metrics import accuracy_score

X_train = np.load("X_train_kaggle.npy")
XX_test = np.load("X_test_kaggle.npy")
y_train_data = np.genfromtxt("groups.csv", delimiter=",", dtype=[("id",np.uint),("group_id",np.uint),("surface","S22")])
y_train = y_train_data["surface"]

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

splitter = model_selection.GroupShuffleSplit(test_size=0.2)
X_train, X_test, y_train, y_test = splitter.split(X_train, y_train, y_train_data["group_id"])

X_train_init = np.array([x.ravel() for x in X_train])
XX_test = np.array([x.ravel() for x in XX_test])

clf = discriminant_analysis.LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

print(accuracy_score(clf.predict(X_train),y_train))

#%%

from sklearn import neighbors, discriminant_analysis, svm, linear_model

classifiers= [neighbors.KNeighborsClassifier(),
              discriminant_analysis.LinearDiscriminantAnalysis(),
              svm.SVC(),
              linear_model.LogisticRegression()]

for classifier in classifiers:
    classifier.fit(X_train,y_train)
    
    print(accuracy_score(y_test, classifier.predict(X_test)))
    
#%%

import xgboost as xgb

dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)


#%%

y_pred = classifier.predict(X_train)
labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        fp.write("%d,%s\n" % (i, label))


#%%