# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:05:45 2019

@author: Joni Seppälä

This is my script for the project work on the course SGN-41007 Pattern Recognition and Machine Learning.
It is also the script our group used to generate ourr submission in the Kaggle competition

    https://www.kaggle.com/c/robotsurface/

The script reads data from csv-files, extracts features out of the data and fits it into many classifiers.
Based on accuracy scores, it either continues or starts the next iteration.

The script takes the most common vote of the classifiers' predictions
and uses them to generate a csv-file submissionXXXX.
Alternatively, the script may generate an average over all CSV-files in its directory.
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

from sklearn.base import clone

def mostVoted(X):
    '''This function returns most voted predictions of classifiers
    '''
    y_preds = []
    for clf in classifiers:
        y_preds.append(clf.predict(X))
    
    y_pred = []
    for i in range(len(y_preds[0])):
        preds = []
        for j in range(len(y_preds)):
            preds.append(y_preds[j][i])
        y_pred.append(np.argmax(np.bincount(preds)))
        
    return y_pred

threshold_min = 0.5 #Minimum threshold; move to next iteration if any classifier has less accuracy
threshold_valid = 0.7 #Create CSV-file if accuracy more than this

K = 0 #Iteration index
K_max = 10000 #Max iterations

'''Iterate the script K_max times, creating csv-files of good predictions.
'''
while K < K_max:

    flag = True #Reset flag  - Flag represents if we should continue to the next iteration
    
    #%% 1. Load data.
    
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
    '''
    X_train1 = np.array([x.ravel() for x in X_train])
    X_test1 = np.array([x.ravel() for x in X_test])
    '''
    #%% 4 (b) The average over the time axis
    
    X_train2 = np.mean(X_train,axis=2)
    X_test2 = np.mean(X_test,axis=2)
    
    X_kaggle_test2 = np.mean(X_kaggle_test,axis=2)
    
    #%% 4 (c) The average and standard deviation over the time axis
    
    ''' Chosen vectorization approach
    '''
    
    X_train = np.std(X_train,axis=2)
    X_train = np.concatenate((X_train2, X_train), axis=1)
    
    X_test = np.std(X_test,axis=2)
    X_test = np.concatenate((X_test2, X_test), axis=1)
    
    #competition testing data
    X_kaggle_test = np.std(X_kaggle_test,axis=2)
    X_kaggle_test = np.concatenate((X_kaggle_test2, X_kaggle_test), axis=1)
    
    #%% A piece of code to find out what features are important (needs to have classifiers set up)
    '''
    importances = []
    for clf in classifiers:
        if hasattr(clf, 'feature_importances_'):
            importances.append(clf.feature_importances_)
    
    mean_importances = []
    for i,importance in enumerate(importances[0]):
        mean_importances.append(np.mean([importances[0][i],importances[1][i],importances[2][i]]))
    '''
    
    #%% Use only 10 most important features (important according to the ensemble methods' method "feature_importances_")
    
    important_features = [0,1,2,3,11,14,15,17,18,19]
    X_train = X_train[:,important_features]
    X_test = X_test[:,important_features]
    X_kaggle_test = X_kaggle_test[:,important_features]
    
    #%% 5. Try different models
    
    classifiers= [#neighbors.KNeighborsClassifier(n_neighbors=1), #NNs don't learn the concept very well
                  #neighbors.KNeighborsClassifier(n_neighbors=5),
                  discriminant_analysis.LinearDiscriminantAnalysis(),
                  svm.SVC(kernel="linear"),
                  #svm.SVC(kernel="rbf",gamma="auto"), #Having only 1 SVM is reasonable
                  linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial",max_iter=2000),
                  RandomForestClassifier(n_estimators=1000),
                  #AdaBoostClassifier(), #Produced so bad results I let it go
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
    
    #%% Create a new instance of each classifier, fit it with data and get the accuracy
    
    for i, clf in enumerate(classifiers):
        
        clf = clone(clf)
        classifiers[i] = clf
        
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(clf_names[i],": %.3f%%" % (100 * accuracy))
        
        if accuracy < threshold_min:
            print("\nBelow accuracy threshold, moving to next iteration...\n"+"_"*53+"\n")
            flag = False
            break
    
    if not flag:
        continue #Next iteration
      
    #%% Take the most voted predictions of classifiers and get their accuracy
    
    y_pred = mostVoted(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    print("\nMost common vote:",100*accuracy,"%\n")
    
    if accuracy > threshold_valid:
        #Save file if accuracy of most voted predictions is more than 70%
        #NOTE: Only one of the following three methods of creating y_pred should be active
        print("Accuracy more than 70%, saving file...\n")
        
        #%% METHOD 1: Use a chosen classifier and make y_pred with it
        '''
        # RandomForest chosen as the classifier
        classifier = classifiers[6]
        y_pred = classifier.predict(X_kaggle_test)
        '''
        #%% METHOD 2: Take the most voted predictions of classifiers and make y_pred out of them
        
        y_pred = mostVoted(X_kaggle_test)
        
        #%% METHOD 3: Most voted predictions of already generated csv-files
        '''
        path = '..\\submissions'
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
        '''
        #%% Create a submission file
        
        labels = list(le.inverse_transform(y_pred))
        
        path = "..\\submissions"
        os.chdir(path)
        
        name = "submission"+str(accuracy/10)[3:7]+".csv"
        
        with open(name, "w") as fp:
            fp.write("# Id,Surface\n")
            for i, label in enumerate(labels):
                #print (str(label)[2:-1])
                fp.write("%d,%s\n" % (i, str(label)[2:-1]))
        
        #%%
        print("Next iteration...\n"+"_"*17+"\n")
        
    else:
        print("Accuracy less than 70%, next iteration...\n"+"_"*41+"\n")

    K += 1









