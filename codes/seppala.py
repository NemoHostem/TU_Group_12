# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:05:45 2019

@author: jonis
"""

#%% 1. Load data

import os
import glob

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn import neighbors, svm, linear_model, discriminant_analysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

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
train, test = next(GroupShuffleSplit(n_splits=1,test_size=0.1).split(X_train, groups=groups))
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

#%% Use only 10 most important features (important according to the ensemble methods method "feature_importances_")

important_features = [0,1,2,3,11,14,15,17,18,19]
X_train = X_train[:,important_features]
X_test = X_test[:,important_features]
X_kaggle_test = X_kaggle_test[:,important_features]

#%% 5. Try different models

classifiers= [#neighbors.KNeighborsClassifier(n_neighbors=1),
              #neighbors.KNeighborsClassifier(n_neighbors=5),
              discriminant_analysis.LinearDiscriminantAnalysis(),
              svm.SVC(kernel="linear"),
              #svm.SVC(kernel="rbf",gamma="auto"),
              linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial",max_iter=2000),
              RandomForestClassifier(n_estimators=1000),
              #AdaBoostClassifier(),
              ExtraTreesClassifier(n_estimators=1000),
              GradientBoostingClassifier(),
              XGBClassifier(n_estimators=1000)]

clf_names = [#"1-NN",
             #"5-NN",
             "LDA",
             "Linear SVC",
             #"RBF SVC",
             "Logistic Regression",
             "RandomForest",
             #"AdaBoost",
             "Extra Trees",
             "GB-Trees",
             "XGBClassifier"]

for i, clf in enumerate(classifiers):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf_names[i],": %.3f%%" % (100 * accuracy_score(y_test, y_pred)))
  
#%%
    
y_preds = []
for clf in classifiers:
    y_preds.append(clf.predict(X_test))

y_pred = []
for i in range(len(y_preds[0])):
    preds = []
    for j in range(len(y_preds)):
        preds.append(y_preds[j][i])
    y_pred.append(np.argmax(np.bincount(preds)))

print("\nMost common vote:",100*accuracy_score(y_test,y_pred),"%\n")

#%% A piece of code to find out what features are important
'''
importances = []
for clf in classifiers:
    if hasattr(clf, 'feature_importances_'):
        importances.append(clf.feature_importances_)

mean_importances = []
for i,importance in enumerate(importances[0]):
    mean_importances.append(np.mean([importances[0][i],importances[1][i],importances[2][i]]))
'''
#%% Most voted prediction of classifiers

#''' RandomForest chosen as the classifier
#'''
#classifier = classifiers[6]
#y_pred = classifier.predict(X_kaggle_test)

y_preds = []
for clf in classifiers:
    y_preds.append(classifiers[-2].predict(X_kaggle_test))

y_pred = []
for i in range(len(y_preds[0])):
    preds = []
    for j in range(len(y_preds)):
        preds.append(y_preds[j][i])
    y_pred.append(np.argmax(np.bincount(preds)))

#%% Most voted predictions of already generated csv-files

path = '.\\'
extension = 'csv'

os.chdir(path)
csv_files = [i for i in glob.glob('*.{}'.format(extension))]

submissions = []
for csv_file in csv_files:
    submission_data = np.genfromtxt(csv_file, delimiter=",", dtype=[("id",np.uint),("surface","S22")])
    submission = submission_data["surface"]
    submissions.append(le.transform(submission))
    
y_pred = []
for i in range(len(submissions[0])):
    preds = []
    for j in range(len(submissions)):
        preds.append(submissions[j][i])
    y_pred.append(np.argmax(np.bincount(preds)))

#%% Create submission file

labels = list(le.inverse_transform(y_pred))
with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, label in enumerate(labels):
        #print (str(label)[2:-1])
        fp.write("%d,%s\n" % (i, str(label)[2:-1]))

#%%













