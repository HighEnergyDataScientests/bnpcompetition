#!/usr/bin/python
###################################################################################################################
### This code is developed by HighEnergyDataScientests Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import operator

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import time
import os


### Controlling Parameters
output_col_name = "target"
test_col_name = "PredictedProb"
enable_feature_analysis = 1
id_col_name = "ID"
num_iterations = 5


def ceate_feature_map(features,featureMapFile):
    outfile = open(featureMapFile, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
def train_model(features,params,num_boost_round,test_size):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print "########################## Time Stamp ==== " + timestamp
    
    print("## Train a XGBoost model")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
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
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=150, verbose_eval=True)


    if enable_feature_analysis == 1:
        print("## Creating Feature Importance Map")
        featureMapFile = '../feature_analysis/xgb_' + timestr +'.fmap'
        ceate_feature_map(features,featureMapFile)
        importance = gbm.get_fscore(fmap=featureMapFile)
        importance = sorted(importance.items(), key=operator.itemgetter(1))

        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 12))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        fig_featp = featp.get_figure()
        fig_featp.savefig('../feature_analysis/feature_importance_xgb_' + timestr + '.png',bbox_inches='tight',pad_inches=1)
        df.to_csv('../feature_analysis/feature_importance_xgb_' + timestr + '.csv')

    print("## Predicting test data")
    preds = gbm.predict(xgb.DMatrix(test[features]),ntree_limit=gbm.best_ntree_limit)
    test[test_col_name] = preds
    test[[id_col_name,test_col_name]].to_csv("../predictions/pred_" + timestr + ".csv", index=False)
    print("Best Score for this run : " + str(gbm.best_score))
    models_predictions["run_"+timestr] = preds
    models_predictions["run_"+timestr].shift(1)
    models_predictions.iloc[0,-1] = gbm.best_score
    models_predictions.to_csv(models_predictions_file, index=False)
    

def predict_missing_data_for_column(features,missing_column,params,num_boost_round,test_size,train_file_name,test_file_name):
    print("## Train a XGBoost model for filling missing column : " + str(missing_column))
    
    X_missing_data_train = train[train[missing_column].isnull()]
    X_missing_data_test = test[test[missing_column].isnull()]
    
    X_data_train = train[np.isfinite(train[missing_column])]
    X_data_test = test[np.isfinite(test[missing_column])]
    
    X_data = pd.concat([X_data_train,X_data_test])
    X_data = X_data.iloc[np.random.permutation(len(X_data))]
        
    #print(X_missing_data[missing_column])
    #print(X_data[missing_column])
    
    X_train, X_valid = train_test_split(X_data, test_size=test_size)
    
    y_train = X_train[missing_column]
    y_valid = X_valid[missing_column]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
    fgbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

    print("## Predicting missing data for column : " + str(missing_column))
    
    if not X_missing_data_train.empty:
        fpreds = fgbm.predict(xgb.DMatrix(X_missing_data_train[features]),ntree_limit=fgbm.best_ntree_limit)
        train.loc[train[missing_column].isnull(),missing_column] = fpreds
    
    if not X_missing_data_test.empty:
        fpreds = fgbm.predict(xgb.DMatrix(X_missing_data_test[features]),ntree_limit=fgbm.best_ntree_limit)
        test.loc[test[missing_column].isnull(),missing_column] = fpreds
    
    train.to_csv(train_file_name, index=False)
    test.to_csv(test_file_name, index=False)
    
    print("##########################################################################################################################")
    
def predict_missing_data(drop_columns):
    print("## Train a XGBoost models to fill missing columns............")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    col_nan_count_train = train.isnull().sum()
    col_nan_count_test = test.isnull().sum()
    #print col_nan_count
    
    features_no_missing_data_train_all = list(col_nan_count_train[col_nan_count_train == 0].index.ravel())
    features_no_missing_data_train = list(set(features_no_missing_data_train_all) - set(drop_columns))
    
    features_no_missing_data_test_all = list(col_nan_count_test[col_nan_count_test == 0].index.ravel())
    features_no_missing_data_test = list(set(features_no_missing_data_test_all) - set(drop_columns))
    
    print("    Features no missing data in training data : " + str(features_no_missing_data_train))
    print("    Features no missing data in testing data : " + str(features_no_missing_data_test))
    
    features_no_missing_data_both = list(set(features_no_missing_data_train) & set(features_no_missing_data_test))
    
    print("    Features no missing data in both data sets : " + str(features_no_missing_data_both))
    
    features_missing_data_train = list(set(list(col_nan_count_train.index.ravel())) - set(features_no_missing_data_both))
    features_missing_data_test = list(set(list(col_nan_count_test.index.ravel())) - set(features_no_missing_data_both))
    
    features_missing_data_both_not_filtered = list(set(features_missing_data_train) | set(features_missing_data_test))
    
    features_missing_data_both = list(set(features_missing_data_both_not_filtered) - set(drop_columns))
    
    print("    Features missing data in training : " + str(features_missing_data_train))
    print("    Features missing data in testing : " + str(features_missing_data_test))
    
    print("    Features missing data in both : " + str(features_missing_data_both))
    
    params = {"objective": "reg:linear",
              "eta": 0.3,
              "nthread":3,
              "max_depth": 10,
              "subsample": 0.75,
              "colsample_bytree": 0.8,
              "eval_metric": "rmse",
              "n_estimators": 20,
              "silent": 1,
              "seed": 23423
              }
    num_boost_round = 200
    test_size = 0.05
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    
    train_filed_file_name = '../intermediate_data/train_filed_' + timestr + '.csv'
    test_filed_file_name = '../intermediate_data/test_filed_' + timestr + '.csv'
    
    number_of_featuers = len(features_missing_data_both)
    count = 1
    for missing_feature in features_missing_data_both:
        print("### Feature Number " + str(count) + " of " + str(number_of_featuers)) 
        predict_missing_data_for_column(features_no_missing_data_both,missing_feature,params,num_boost_round,test_size,train_filed_file_name,test_filed_file_name)
        count +=1
    
    col_nan_count_train = train.isnull().sum()
    col_nan_count_test = train.isnull().sum()
    
    print col_nan_count_train
    print col_nan_count_test
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    
    print("Columns Still have nan : ")
    print col_nan_count_train[col_nan_count_train != 0]
    print col_nan_count_test[col_nan_count_test != 0]
    train.to_csv(train_filed_file_name, index=False)
    test.to_csv(test_filed_file_name, index=False)



timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Start Time Stamp ==== " + timestamp)
print("## Loading Data")
models_predictions_file = "../predictions/models_predictions.csv"
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')


if os.path.isfile(models_predictions_file):
    models_predictions = pd.read_csv(models_predictions_file)
else:
    models_predictions = pd.DataFrame()
 
timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Time Stamp ==== " + timestamp)
print("## Data Processing")
train = train.drop(id_col_name, axis=1)

#train = train.fillna(-1)
#test = test.fillna(-1)

timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Time Stamp ==== " + timestamp)
print("## Data Encoding")
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
print("Features: ", features)

print train

#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit(train[features])
#train[features] = imp.transform(train[features])
#test[features] = imp.transform(test[features])

timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Time Stamp ==== " + timestamp)
print("## Training")
numPos = len(train[train[output_col_name] == 1])
numNeg = len(train[train[output_col_name] == 0])
scaleRatio = float(numNeg) / float(numPos)
print("Number of postive " + str(numPos) + " , Number of negative " + str(numNeg) + " , Ratio Negative to Postive : " , str(scaleRatio))


drop_columns = ["ID","target"]

predict_missing_data(drop_columns)


params = {"objective": "binary:logistic",
          "eta": 0.05,
          "nthread":3,
          "max_depth": 6,
          "subsample": 0.67,
          "colsample_bytree": 0.9,
          "eval_metric": "logloss",
          "n_estimators": 100,
          "silent": 1,
          "seed": 93425
          }
num_boost_round = 1000
test_size = 0.05
train_model(features,params,num_boost_round,test_size)

timestamp = time.strftime("%Y%m%d-%H%M%S")
print "########################## End Time Stamp ==== " + timestamp
    
