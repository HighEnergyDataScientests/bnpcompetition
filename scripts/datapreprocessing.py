#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics team on kaggle
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################
#
############################### This is datapreprocessing module ##################################################

import pandas as pd
import numpy as np
#import os

from sklearn import preprocessing
from sklearn.preprocessing import Imputer

from sklearn.cross_validation import train_test_split

def data_processing(models_predictions_file,train,test, output_col_name,test_col_name,id_col_name,test_size_test,test_size_val):

   # data processing
   print("## Data Processing")
   #if os.path.isfile(models_predictions_file):
   #    models_predictions = pd.read_csv(models_predictions_file)
   #else:
   #    models_predictions = pd.DataFrame()


   # encoding data
   print("## Encoding Data")
   train = train.drop(id_col_name, axis=1)
   for f in train.columns:
       if train[f].dtype=='object':
           print(f)
           lbl = preprocessing.LabelEncoder()
           lbl.fit(list(train[f].values) + list(test[f].values))
           train[f] = lbl.transform(list(train[f].values))
           test[f] = lbl.transform(list(test[f].values))

   features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
   print("Features: ", features)

   imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
   imp.fit(train[features])
   train[features] = imp.transform(train[features])
   test[features] = imp.transform(test[features])

   y = train[output_col_name]

   # splitting training data into training and cross validation sets
   X_pos = train[train[output_col_name] == 1]
   X_neg = train[train[output_col_name] == 0]

   # Splitting train to X_train and X_test
   X_train_pos, X_test_pos = train_test_split(X_pos, test_size=test_size_test)
   X_train_neg, X_test_neg = train_test_split(X_neg, test_size=test_size_test)

   # Splitting train to X_train and X_valid    
   X_train_pos, X_valid_pos = train_test_split(X_train_pos, test_size=test_size_val)
   X_train_neg, X_valid_neg = train_test_split(X_train_neg, test_size=test_size_val)

   X_test = pd.concat([X_test_pos,X_test_neg])    
   X_train = pd.concat([X_train_pos,X_train_neg])
   X_valid = pd.concat([X_valid_pos,X_valid_neg])

   X_test = X_test.iloc[np.random.permutation(len(X_test))]    
   X_train = X_train.iloc[np.random.permutation(len(X_train))]
   X_valid = X_valid.iloc[np.random.permutation(len(X_valid))]

   y_test = X_test[output_col_name]
   y_train = X_train[output_col_name]
   y_valid = X_valid[output_col_name]

   # creating temp_test array to use to compute predictions with test data set
   temp_test = test.drop(id_col_name, axis=1)

   # deleting first column, which corresponds to output y, from both data sets
   train = train.drop(output_col_name, axis=1)
   X_test = X_test.drop(output_col_name, axis=1)
   X_train = X_train.drop(output_col_name, axis=1)
   X_valid = X_valid.drop(output_col_name, axis=1)

   # scaling data
   print("## Scaling Data")
   train = preprocessing.scale(train)
   X_test = preprocessing.scale(X_test)
   X_train = preprocessing.scale(X_train)
   X_valid = preprocessing.scale(X_valid)
   temp_test = preprocessing.scale(temp_test)

   return train, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test