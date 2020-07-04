"""
=====================
Evaluating Deep Learning classifiers on Static Features

=====================
This uses Pytorch's dataloader functionality

Run command: 
$ python dl.py simpleAnn ../csv_files/scuf.csv.gz ../csv_files/scores_msoft20.csv
$ python dl.py complexAnn ../csv_files/scuf.csv.gz ../csv_files/scores_msoft20.csv
$ python dl.py Cnn ../csv_files/scuf.csv.gz ../csv_files/scores_msoft20.csv
$ python dl.py Rnn ../csv_files/scuf.csv.gz ../csv_files/scores_msoft20.csv
"""

print(__doc__)

import os, sys
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import Dataset, DataLoader

# Cuda for pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]

        # Load data and get label
        X = self.data[index]
        y = self.labels[index]

        return X, y

# count param function that shows the specifications of the model
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

# simple ANN (sANN) 
class SimpleAnn(nn.Module):
    def __init__(self, in_features, h1=256, h2=256, out_features=2):
       
        super().__init__()
        # fc -- fully connected layer
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
       
    def forward(self,x): # propagation method
        # use ReLu activation funciton 
        x = F.relu(self.fc1(x)) # feed forward 
        x = F.relu(self.fc2(x)) # feed forward
        x = self.out(x)

        return x

# complex ANN (cANN)
class ComplexAnn(nn.Module):
    def __init__(self, in_features, out_features=2, layers=[512,256,128], p=0.5):
        super().__init__()
       
        layerlist = []
       
        for i in layers:
            layerlist.append(nn.Linear(in_features,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.Dropout(p))
            in_features = i
        layerlist.append(nn.Linear(layers[-1],out_features))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x):
        x = self.layers(x)
        return x

# CNN 
class ConvolutionalNetwork(nn.Module):
    def __init__(self, in_features, p=0.5):
        super().__init__()
        self.in_features = in_features
        # this computes num features outputted from the two conv layers
        c1 = int(((self.in_features - 2)) / 64) 
        c2 = int((c1-2)/64)
        self.n_conv = int(c2*16)
        
        self.conv1 = nn.Conv1d(1, 16, 3, 1)
        self.conv2 = nn.Conv1d(16, 16, 3, 1)
        self.dp = nn.Dropout(p)
        self.fc3 = nn.Linear(self.n_conv, 2)

    def forward(self, x):
        # shape x for conv 1d op
        x = x.view(-1,1,self.in_features)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x,64,64)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x,64,64)
        x = x.view(-1,self.n_conv)
      
        x = self.dp(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        return x

# RNN 
class LSTMnetwork(nn.Module):
    def __init__(self, in_features, hidden_sz=6, p=0.5, out_features=2):
        super(LSTMnetwork, self).__init__()
        self.hidden_sz = hidden_sz

        self.in_features = in_features

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.in_features, hidden_sz)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_sz, out_features)
      
        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1,1,self.hidden_sz),
                       torch.zeros(1,1,self.hidden_sz))

    def forward(self,x):
        lstm_out, self.hidden = self.lstm(
            x.view(len(x), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(x),-1))
        y_val = F.log_softmax(pred, dim=1)
        return y_val
       

def run_analysis():
    print("Running " + name + " classifier on "+ feature_type + " features...")
    
    scores = {}
    scores['test_recall'] = []
    scores['test_precision'] = []
    scores['test_f1'] = []
    scores['fit_time'] = []

    df = pd.read_csv(inputFile, compression='gzip',index_col=0)

    X = df.dropna(axis=1, how='all') # drop columns (axis=1) with 'all' NaN values
    # get data without label
    X = X.drop('classLabel', axis=1)
    # y = labels
    y = df['classLabel']
    #print(X)
    in_features = len(X.columns) # number of features
    num_instances = len(X) # number of instances

    # convert to numpy arrays
    X = X.values
    y = y.values

    #Stratified cross-validation
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=33)
    eval_cnt = 0  # total number of evaluations = 5 fold * 5 times = 25
    for train_index, test_index in rskf.split(X, y):
        train_start_time = time.time()
        eval_cnt+=1
        print("Evaluation " + str(eval_cnt))
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # convert into tensor
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
       
        # sklearn's dataloader, useful for large dataset
        # it also does data shuffling
        trainDT = CustomDataset(X_train, y_train)
        testDT = CustomDataset(X_test, y_test)
        trainloader = DataLoader(trainDT, batch_size=500, shuffle=True)
        testloader = DataLoader(testDT, batch_size=500, shuffle=False)

        if name.lower() == 'complexann':
            model = ComplexAnn(in_features=in_features)
        elif name.lower() == 'cnn':
            model = ConvolutionalNetwork(in_features=in_features)
        elif name.lower() == 'rnn':
            model = LSTMnetwork(in_features=in_features)
        else:  # default
            model = SimpleAnn(in_features=in_features)

        # define loss function to meaure error
        criterion = nn.CrossEntropyLoss()  
      
        # define optimizer to update weights and biases
        # learning rate, lr, is set to 0.001.
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
       
        # Train the model
        for _ in range(epochs):
            # using data_loader 
            for i, (data, labels) in enumerate(trainloader):
                # Forward and get a prediction
                # x is the training data which is X_train
                if name.lower() == "rnn":
                    model.hidden = (torch.zeros(1,1,model.hidden_sz),
                        torch.zeros(1,1,model.hidden_sz))
              
                y_pred = model.forward(data)
              
                # compute loss/error by comparing predicted out vs acutal labels
                loss = criterion(y_pred, labels)
                #losses.append(loss)
                
                if i%10==0:  # print out loss at every 10 epoch
                    print(f'epoch {i} and loss is: {loss}')
                
                #Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validate the model
        train_end_time = time.time()
        fit_time = train_end_time - train_start_time
        correct = 0
        tp=0; tn=0; fp=0; fn=0
        with torch.no_grad(): 
            for _, (data, labels) in enumerate(testloader):
                y_val = model(data)
                predicted = torch.max(y_val.data,1)[1]
                tp,tn,fp,fn,correct = compute_measures(predicted, labels, tp, tn, fp, fn, correct)
            print(f'We got {correct} correct!')

        compute_scores(tp,tn,fp,fn, fit_time, scores)

    # save the model
    modelName = "../savedModels/" + name.replace(" ","") + "_" + feature_type + "_cv" + str(cv) + '_dl.pt'
    torch.save(model.state_dict(),modelName)

    # format output
    convertscores2numpy(scores)
    #print(scores)
    return num_instances, in_features, scores

def compute_measures(predicted, labels, tp, tn, fp, fn, correct):
    print("Predicted:\t" + str(predicted))
    print("Actual:\t\t" + str(labels))
    #print(p2)
    for i, y_pred in enumerate(predicted):
        tp,tn,fp,fn = compute_confusionmatrix(y_pred, labels[i], tp, tn, fp, fn)
        
    correct += (predicted==labels).sum()

    return tp, tn, fp, fn, correct
    
def compute_confusionmatrix(y_pred, y_act,tp, tn, fp, fn):
    #print("predicted: " + str(y_pred))
    #print("actual: " + str(y_act))
    if y_act == 1:
        if y_act == y_pred:
            tp+=1
            #print("true positive")
        else:
            fn+=1
            #print("false negative")
    else: 
        if y_act == y_pred:
            tn+=1
            #print("true negative")
        else:
            fp+=1
            #print("false positive")
    return tp, tn, fp, fn

def compute_scores(tp,tn,fp,fn, fit_time, scores):
    try: 
        recall = tp / (tp+fp)
    except: # handle ZeroDivisionError
        recall = 0
    try:
        precision = tp / (tp+fn)
    except:
        precision = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0

    scores['test_recall'].append(recall)
    scores['test_precision'].append(precision)
    scores['test_f1'].append(f1)
    scores['fit_time'].append(fit_time)


def convertscores2numpy(scores):
    scores['test_recall'] = np.array(scores['test_recall'])
    scores['test_precision'] = np.array(scores['test_precision'])
    scores['test_f1'] = np.array(scores['test_f1'])
    scores['fit_time'] = np.array(scores['fit_time'])

if __name__ == '__main__':
    start_time = time.time()

    try:
        name = sys.argv[1]
    except:
        name = 'simpleann'  # default dl model 

    try: 
        inputFile = sys.argv[2]
    except:
        #inputFile = '../csv_files/smsf.csv.gz'  # static-sequence features
        inputFile = '../csv_files/scuf.csv.gz'  # static-use features
        #inputFile = '../csv_files/dmsf.csv.gz'  # dynamic-sequence features
        #inputFile = '../csv_files/dcuf.csv.gz'  # dynamic-use features
        #inputFile = '../csv_files/hmsf.csv.gz'  # hybrid-sequence features
        #inputFile = '../csv_files/hcuf.csv.gz'  # hybrid-use features
        
    try:
        outFile = sys.argv[3]
    except:
        outFile = '../csv_files/scores_msoft20.csv'

    # determine type of features that are being used
    idx = inputFile.rfind('/') + 1
    feature_type = inputFile[idx:]

    epochs = 30 
    # parameters for cross validation -> 5fold cv
    cv = 5
    n_splits = cv
    n_repeats = cv

    n_instances, n_features, scores = run_analysis()

    note = ""
    #result = 'Classifier, Type, Instances, Features, Recall (avg), Recall (std dev), Precision (avg), Precision (std dev), F1 (avg), F1 (std dev), Training Time (avg), Training Time (std dev), CV, Duration, Note, Date\n'
    result = ""
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

    result += str(n_repeats) +  ","
    result += f'{(time.time() - start_time)/3600:.2f},'
    result += note + ","
    result += str(datetime.datetime.now()) + '\n'

    print(result)
    f = open(outFile, "a+")
    f.write(result)
    f.close()