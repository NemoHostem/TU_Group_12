# -*- coding: utf-8 -*-
"""
Created on Sun Feb 3 17:18:47 2019
Edited on Sun Feb 10 11:15:27 2019

@author: Matias
"""

#%% 1. Load data

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn import discriminant_analysis
from sklearn.metrics import accuracy_score
from sklearn import neighbors, svm, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

path = ''

X_train_orig = np.load(path+"X_train_kaggle.npy")
X_kaggle_test = np.load(path+"X_test_kaggle.npy")
y_train_data = np.genfromtxt(path+"groups.csv", delimiter=",", dtype=[("id",np.uint),("group_id",np.uint),("surface","S22")])
y_train_orig = y_train_data["surface"]

#%% 2. Create an index of class names.

le = preprocessing.LabelEncoder()
le.fit(y_train_orig)
y_train = le.transform(y_train_orig)

#%% 3. Split to training and testing.
groups = y_train_data["group_id"]
res = np.empty([10,100])

for r in range(100):

    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=r).split(X_train_orig, groups=groups))
    X_train, X_test, y_train, y_test = X_train_orig[train], X_train_orig[test], y_train_orig[train], y_train_orig[test]


    X_train1 = np.array([x.ravel() for x in X_train])
    X_test1 = np.array([x.ravel() for x in X_test])


    X_train2 = np.mean(X_train,axis=2)
    X_test2 = np.mean(X_test,axis=2)


    X_train = np.std(X_train,axis=2)
    X_train = np.concatenate((X_train2, X_train), axis=1)
    
    X_test = np.std(X_test,axis=2)
    X_test = np.concatenate((X_test2, X_test), axis=1)
    
    

    classifiers= [neighbors.KNeighborsClassifier(n_neighbors=1),
                  neighbors.KNeighborsClassifier(n_neighbors=5),
                  discriminant_analysis.LinearDiscriminantAnalysis(),
                  svm.SVC(kernel="linear"),
                  svm.SVC(kernel="rbf",gamma="auto"),
                  linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial",max_iter=2000),
                  RandomForestClassifier(n_estimators=1000),
                  AdaBoostClassifier(),
                  ExtraTreesClassifier(n_estimators=1000),
                  GradientBoostingClassifier()]
    
    classifiers_names = ["1-NN","5-NN","LDA","Linear SVC","RBF SVC","Logistic Regression","RandomForest","AdaBoost","Extra Trees","GB-Trees"]
    
    for i, classifier in enumerate(classifiers):
        classifier.fit(X_train,y_train)
        res[i][r] = (accuracy_score(y_test, classifier.predict(X_test)))
        print(classifiers_names[i]+": "+str(100*accuracy_score(y_test, classifier.predict(X_test)))+" %")
    
#%% Print mean, min and max of each classifier
        
for n, name in enumerate(classifiers_names):
    print("Mean of " + name + ": " + str(100*np.mean(res[n][:])) + " %")
    print("Min of " + name + ": " + str(100*np.min(res[n][:])) + " %")
    print("Max of " + name + ": " + str(100*np.max(res[n][:])) + " %")

'''
import xgboost as xgb

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
'''

#%% Create submission file

''' RandomForest chosen as the classifier

classifier = classifiers[6]

y_pred = classifier.predict(X_kaggle_test)
labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        print (str(label)[2:-1])
        fp.write("%d,%s\n" % (i, str(label)[2:-1]))

'''
#%%