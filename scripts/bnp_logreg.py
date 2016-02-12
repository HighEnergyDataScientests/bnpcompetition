#!/usr/bin/python
###################################################################################################################
### This code is developed by HighEnergyDataScientests Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

import matplotlib
matplotlib.use("Agg") #Needed to save figures
import time
import os
#import math

# Logistic Regression
from sklearn.linear_model import LogisticRegression

### Controlling Parameters
output_col_name = "target"
test_col_name = "PredictedProb"
enable_feature_analysis = 1
id_col_name = "ID"
num_iterations = 5
test_size = 0.1

# load the datasets
print("## Loading Data")
models_predictions_file = "../predictions/models_predictions.csv"
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')

# data processing and encoding
if os.path.isfile(models_predictions_file):
    models_predictions = pd.read_csv(models_predictions_file)
else:
    models_predictions = pd.DataFrame()
 

print("## Data Processing")
train = train.drop(id_col_name, axis=1)

print("## Data Encoding")
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# encoding training data set
features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
print("Features: ", features)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train[features])
train[features] = imp.transform(train[features])
test[features] = imp.transform(test[features])

# encoding test data set
X_pos = train[train[output_col_name] == 1]
X_neg = train[train[output_col_name] == 0]
    
X_train_pos, X_valid_pos = train_test_split(X_pos, test_size=test_size)
X_train_neg, X_valid_neg = train_test_split(X_neg, test_size=test_size)
    
X_train = pd.concat([X_train_pos,X_train_neg])
X_valid = pd.concat([X_valid_pos,X_valid_neg])
    
X_train = X_train.iloc[np.random.permutation(len(X_train))]
X_valid = X_valid.iloc[np.random.permutation(len(X_valid))]
    
y_train = X_train[output_col_name]
y_valid = X_valid[output_col_name]


# fit a logistic regression model to the data
print("## Training")
model = LogisticRegression()
model.fit(X_train, y_train)
timestr = time.strftime("%Y%m%d-%H%M%S")

# make predictions
print("## Making Predictions")

p = model.predict_proba(test)
p_log = model.predict_log_proba(test)

#loglosss = -sum(y_train*p_log[:,1] + (1.0 - y_train)*math.log10(1.0 - p[:,1]))
print("Score = ", p_log)

# Producing output of predicted probabilities and writing it into a file
test[test_col_name] = p[:,1]
test[[id_col_name,test_col_name]].to_csv("../predictions/pred_logreg_" + timestr + ".csv", index=False)
