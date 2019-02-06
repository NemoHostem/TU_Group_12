# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:05:45 2019

@author: jonis
"""

#%% 1. Load data

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn import discriminant_analysis
from sklearn.metrics import accuracy_score

path = '../data/'

X_train = np.load(path+"X_train_kaggle.npy")
X_kaggle_test = np.load(path+"X_test_kaggle.npy")
y_train_data = np.genfromtxt(path+"groups.csv", delimiter=",", dtype=[("id",np.uint),("group_id",np.uint),("surface","S22")])
y_train = y_train_data["surface"]

#%% 2. Create an index of class names.

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

#%% 3. Split to training and testing.

groups = y_train_data["group_id"]
train, test = next(GroupShuffleSplit(n_splits=1,test_size=0.2).split(X_train, groups=groups))
X_train, X_test, y_train, y_test = X_train[train], X_train[test], y_train[train], y_train[test]

#%% 4 (a) Straightforward reshape

X_train1 = np.array([x.ravel() for x in X_train])
X_test1 = np.array([x.ravel() for x in X_test])

#%% (b) The average over the time axis

X_train2 = np.mean(X_train,axis=2)
X_test2 = np.mean(X_test,axis=2)

X_kaggle_test2 = np.mean(X_kaggle_test,axis=2)

#%% (c) The average and standard deviation over the time axis

''' Chosen vectorization approach
'''

X_train = np.std(X_train,axis=2)
X_train = np.concatenate((X_train2, X_train), axis=1)

X_test = np.std(X_test,axis=2)
X_test = np.concatenate((X_test2, X_test), axis=1)

#competition testing data
X_kaggle_test = np.std(X_kaggle_test,axis=2)
X_kaggle_test = np.concatenate((X_kaggle_test2, X_kaggle_test), axis=1)

#%% LDA test

'''
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

print(accuracy_score(lda.predict(X_test),y_test))
'''

#%% 5. Try different models

from sklearn import neighbors, svm, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

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
    print(classifiers_names[i]+": "+str(100*accuracy_score(y_test, classifier.predict(X_test)))+" %")
    
#%% Experiment with xgboost; still don't quite understand how it should work

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
'''
classifier = classifiers[6]

y_pred = classifier.predict(X_kaggle_test)
labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        print (str(label)[2:-1])
        fp.write("%d,%s\n" % (i, str(label)[2:-1]))


#%%