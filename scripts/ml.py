"""
=====================
Classifiers comparison on Static Features
A comparison of several classifiers in scikit-learn.
=====================
This uses sklearn's built-in cross validation method

Run command: 
$ python ml.py ../csv_files/scuf.csv.gz ../csv_files/scores_msoft20.csv
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
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from joblib import dump, load
 
def run_analysis():
    # read features file
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

    scoring = ['precision', 'recall', 'f1']
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        start_time = time.time()
        result = ""
        note = ""
        print("Running " + name + " classifier on "+ feature_type + " features...")
      
        filename = "../savedModels/" + name.replace(" ","") + "_" + feature_type + "_cv" + str(cv) + '_ml.joblib'
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        dump(clf, filename)

        #print(scores)
        print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std() * 2))
        print("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std() * 2))
        print("F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))
        print("Fit_time: %0.2f (+/- %0.2f)" % (scores['fit_time'].mean(), scores['fit_time'].std() * 2))

        #result = 'Classifier, Type, Instances, Features, Recall (avg), Recall (std dev), Precision (avg), Precision (std dev), F1 (avg), F1 (std dev), Training Time (avg), Training Time (std dev), CV, Duration, Note, Date\n'
        result += name + "," + feature_type +","
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

        f = open(outFile, "a+")
        f.write(result)
        f.close()
    
if __name__ == '__main__':
   
    try: 
        inputFile = sys.argv[1]
    except:
        #inputFile = '../csv_files/smsf.csv.gz'  # static-sequence features
        inputFile = '../csv_files/scuf.csv.gz'  # static-use features
        #inputFile = '../csv_files/dmsf.csv.gz'  # dynamic-sequence features
        #inputFile = '../csv_files/dcuf.csv.gz'  # dynamic-use features
        #inputFile = '../csv_files/hmsf.csv.gz'  # hybrid-sequence features
        #inputFile = '../csv_files/hcuf.csv.gz'  # hybrid-use features
       
    try:
        outFile = sys.argv[2]
    except:
        outFile = '../csv_files/scores_msoft20.csv'

    # determine type of features that are being used
    idx = inputFile.rfind('/') + 1
    feature_type = inputFile[idx:]

    # define classifiers 
    names = ["Nearest Neighbors", "Linear SVM", 
            "Decision Tree", "Random Forest", "AdaBoost",
            "Naive Bayes", "Logistic Regression"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        LogisticRegressionCV()]

    cv = 5  # five-fold cross validation

    run_analysis()

    