#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics team on kaggle
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################
#
############################### This is BNP_main script which calls to functions from modules #####################

import time
import pandas as pd

# Custom modules
#from datapreprocessing import data_processing
from data_preproc_XGBoost import data_proc_XGBoost
from two_layer_training import First_layer_classifiers, Second_layer_ensembling

# load the datasets
print("## Loading Data")
models_predictions_file = "../predictions/models_predictions.csv"
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')

### Controlling Parameters
output_col_name = "target"
test_col_name = "PredictedProb"
id_col_name = "ID"
test_size_test = 0.2
test_size_val = test_size_test/(1.0 - test_size_test)

## Running data preprocessing
#X, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test = data_processing(models_predictions_file,train,test, output_col_name,test_col_name,id_col_name,test_size_test,test_size_val)

X, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test = data_proc_XGBoost(models_predictions_file,train,test, output_col_name,test_col_name,id_col_name,test_size_test,test_size_val)

p_valid, p_test, p_ttest_t = First_layer_classifiers(X, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test)

yt_out = Second_layer_ensembling(p_valid, p_test, y_valid, y_test, p_ttest_t)

# Producing output of predicted probabilities and writing it into a file
timestamp = time.strftime("%Y%m%d-%H%M%S")

test[test_col_name] = yt_out[:,1]
test[[id_col_name,test_col_name]].to_csv("../predictions/2layer_yt_out" + timestamp + ".csv", index=False)