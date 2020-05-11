"""
=====================
Classifiers comparison on Static Features
A comparison of several classifiers in scikit-learn.
=====================
This uses sklearn's built-in cross validation method

Run command: 
$ python static_ml.py ../csv_files/smsf.csv.gz ../csv_files/scores.csv

"""
print(__doc__)

import os, sys
import time
import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
import pickle
from joblib import dump, load
 

def run_analysis():
    # use this if inputFile is a uncompressed csv file
    # df = pd.read_csv(inputFile)
   
    # use this if there are multiple inputFiles
    #df1 = pd.read_csv(inputFile1, compression='gzip', error_bad_lines=False, index_col=0)
    #df2 = pd.read_csv(inputFile2, compression='gzip', error_bad_lines=False, index_col=0)
    #frames = [df1, df2]
    #df = pd.concat(frames, ignore_index=True)

    # use this for gzip csv files
    df = pd.read_csv(inputFile, compression='gzip',index_col=0)
    
    X = df.dropna(axis=1, how='all') # drop columns (axis=1) with 'all' NaN values
    # get data without label
    X = X.drop('classLabel', axis=1)
    # y = labels
    y = df['classLabel']

    n_features = len(X.columns) # number of features
    n_instances = len(X) # number of instances

    # convert to numpy arrays
    X = X.values
    y = y.values

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=33)

    scoring = ['precision', 'recall', 'f1']
    result = ""
    note = ""
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        start_time = time.time()
        print(name)
        #clf.fit(X_train, y_train)
        #score = clf.score(X_test, y_test)
        filename = name.replace(" ","") + str(cv) + 'staticmlModel.joblib'
        #clfL = load(filename) 
        #scores = cross_validate(clfL, X, y, cv=cv, scoring=scoring)
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        dump(clf, filename)
        
        #print(scores)
        print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std() * 2))
        print("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std() * 2))
        print("F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))
        print("Fit_time: %0.2f (+/- %0.2f)" % (scores['fit_time'].mean(), scores['fit_time'].std() * 2))

        #result = 'Classifier, Type, Instances, Features, Recall (avg), Recall (std dev), Precision (avg), Precision (std dev), F1 (avg), F1 (std dev), Training Time (avg), Training Time (std dev), CV, Duration, Note, Date\n'
        result += name + ",Static,"
        result += str(n_instances) +  ","
        result += str(n_features) + ","
        result += "%0.5f" % (scores['test_recall'].mean()) + ","
        result += "%0.5f" % (scores['test_recall'].std() * 2) + ","
        result += "%0.5f" % (scores['test_precision'].mean()) + ","
        result += "%0.5f" % (scores['test_precision'].std() * 2) + ","
        result += "%0.5f" % (scores['test_f1'].mean()) + ","
        result += "%0.5f" % (scores['test_f1'].std() * 2) + ","
        result += "%0.5f" % (scores['fit_time'].mean()) + ","
        result += "%0.5f" % (scores['fit_time'].std() * 2) + ","

        result += str(cv) +  ","
        result += f'{(time.time() - start_time)/3600:.2f},'
        result += note + ","
        result += str(datetime.datetime.now()) + '\n'
    return result
   

if __name__ == '__main__':
    #start_time = time.time()

    try: 
        inputFile = sys.argv[1]
    except:
        inputFile = '../csv_files/smsf.csv.gz'
        
        
    try:
        outFile = sys.argv[2]
    except:
        outFile = '../csv_files/scores.csv'

    # define classifiers 
    names = ["Nearest Neighbors", "Linear SVM", 
            "Decision Tree", "Random Forest", "AdaBoost",
            "Naive Bayes", "Logistic Regression"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(),
        #MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        LogisticRegressionCV()]

    cv = 5  # five-fold cross validation

    result = run_analysis()

    f = open(outFile, "a+")
    f.write(result)
    f.close()
    