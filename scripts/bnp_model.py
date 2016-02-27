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
import itertools
import random


## Finding columns with first element is less than value
## df.loc[0][df.loc[0] < 0.46]

### Controlling Parameters
output_col_name = "target"
test_col_name = "PredictedProb"
enable_feature_analysis = 1
id_col_name = "ID"
num_iterations = 5
save_limit = 0.46
num_of_trial_comb = 3
exhaustive_grid_search = 0


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


    if enable_feature_analysis == 1 and save_limit > gbm.best_score :
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

    print("Best Score for this run : " + str(gbm.best_score))
    
    print("## Predicting test data")
    if save_limit > gbm.best_score:
        preds = gbm.predict(xgb.DMatrix(test[features]),ntree_limit=gbm.best_ntree_limit)
        test[test_col_name] = preds
        test[[id_col_name,test_col_name]].to_csv("../predictions/pred_" + timestr + ".csv", index=False)
        
        models_predictions["run_"+timestr] = preds
        models_predictions["run_"+timestr].shift(1)
        models_predictions.iloc[0,-1] = gbm.best_score
        models_predictions.to_csv(models_predictions_file, index=False)
    else:
        print("Will not save results as it's higher than our target scoring: " + str(save_limit))
    return gbm.best_score
    

def analyze_data(drop_columns):
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
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    
#    for missing_feature in features_missing_data_both:
#        train_filled_series = train.groupby(features_no_missing_data_both)[missing_feature].mean().reset_index()
#        test_filled_series = test.groupby(features_no_missing_data_both)[missing_feature].mean().reset_index()
#        print(train_filled_series)
#        train[missing_feature].fillna(train_filled_series,inplace=True)
#        test[missing_feature].fillna(test_filled_series,inplace=True)
        
   
#    timestamp = time.strftime("%Y%m%d-%H%M%S")
#    print("########################## Time Stamp ==== " + timestamp)
    
#    print("Columns Still have nan : ")
#    print col_nan_count_train[col_nan_count_train != 0]
#    print col_nan_count_test[col_nan_count_test != 0]



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
# Split data into 2 dataframes, based on the target 
df_pos = train.loc[train[output_col_name] == 1]
df_neg = train.loc[train[output_col_name] == 0]
numPos = len(df_pos)
numNeg = len(df_neg)
scaleRatio = float(numNeg) / float(numPos)
print("Number of postive " + str(numPos) + " , Number of negative " + str(numNeg) + " , Ratio Negative to Postive : " , str(scaleRatio))
num_values = len(train)

print("## Data Encoding")

cat_columns = list(train.select_dtypes(include=['object']).columns)
print("Categorical Features : " + str(cat_columns))

for f in cat_columns:
    print(f)

    # Get the counts of the categories for both pos/neg series
    feat_counts_df = pd.DataFrame(train[f].value_counts(normalize=True))
    #print feat_counts_df
    
    feat_counts_df.fillna(0,inplace=True)

    for v in pd.unique(train[f]):
        if not pd.isnull(v):
            train.loc[(train[f] == v), str(f)+"_pct"] = feat_counts_df.loc[v,f]
            test.loc[(test[f] == v), str(f)+"_pct"] = feat_counts_df.loc[v,f]

    train[str(f)+"_pct"].fillna(0.0,inplace=True)
    test[str(f)+"_pct"].fillna(0.0,inplace=True)

    #print(train[str(f)+"_pct"])
    #print(test[str(f)+"_pct"])

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))


drop_columns = ["ID","target"]
analyze_data(drop_columns)

train["na_count"] = train.count(axis=1)
test["na_count"] = test.count(axis=1)

features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
print("Features: ", features)

print("## Calculating Correlation")
corr_features = features + [output_col_name]
correlation_p = train[corr_features].corr()


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train[features])
train[features] = imp.transform(train[features])
test[features] = imp.transform(test[features])

print("## Creating Random features based on Correlation")
output_cor = correlation_p[output_col_name].sort_values()

most_neg_cor = list(output_cor.index[0:10].ravel())
most_pos_cor = list(output_cor.index[-12:-2].ravel())

print most_neg_cor
print most_pos_cor

exit()


timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Time Stamp ==== " + timestamp)
print("## Training")
eta_list                = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
max_depth_list          = [5, 6, 7, 9, 10, 11 , 13, 14]
subsample_list          = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
colsample_bytree_list   = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
n_estimators_list       = [20, 40, 60, 90, 100, 120, 140]
test_size_list          = [0.05]


grid_search_file = "../predictions/grid_search_file.csv"

if os.path.isfile(grid_search_file):
    grid_search_pd = pd.read_csv(grid_search_file)
else:
    columns = ['eta','max_depth','subsample','colsample_bytree','n_estimators','test_size','result']
    grid_search_pd = pd.DataFrame(columns=columns)

combinations = list(itertools.product(eta_list,max_depth_list,subsample_list,colsample_bytree_list,n_estimators_list,test_size_list))

if exhaustive_grid_search == 0:
    random_sample_to_run = random.sample(combinations,num_of_trial_comb)
    print("## Randomly picked parameters for running training : " + str(random_sample_to_run))

    for t_par in random_sample_to_run:
        eta = t_par[0]
        max_depth = t_par[1]
        subsample = t_par[2]
        colsample_bytree = t_par[3]
        n_estimators = t_par[4]
        test_size = t_par[5]
        params = {"objective": "binary:logistic",
                  "eta": eta,                                              
                  "nthread":3,                                             
                  "max_depth": max_depth,                                  
                  "subsample": subsample,                                  
                  "colsample_bytree": colsample_bytree,                    
                  "eval_metric": "logloss",                                
                  "n_estimators": n_estimators,                            
                  "silent": 1                                              
                  }                                                        
        num_boost_round = 10000
        test_size = test_size
        best_score = train_model(features,params,num_boost_round,test_size)
        grid_search_pd.loc[len(grid_search_pd)] = [eta,max_depth,subsample,colsample_bytree,n_estimators,test_size,best_score]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print "########################## Round Time Stamp ==== " + timestamp

        grid_search_pd.to_csv(grid_search_file, index=False)
else:
    for t_par in combinations:
        eta = t_par[0]
        max_depth = t_par[1]
        subsample = t_par[2]
        colsample_bytree = t_par[3]
        n_estimators = t_par[4]
        test_size = t_par[5]
        params = {"objective": "binary:logistic",
                  "eta": eta,                                              
                  "nthread":3,                                             
                  "max_depth": max_depth,                                  
                  "subsample": subsample,                                  
                  "colsample_bytree": colsample_bytree,                    
                  "eval_metric": "logloss",                                
                  "n_estimators": n_estimators,                            
                  "silent": 1                                              
                  }                                                        
        num_boost_round = 10000
        test_size = test_size
        best_score = train_model(features,params,num_boost_round,test_size)
        grid_search_pd.loc[len(grid_search_pd)] = [eta,max_depth,subsample,colsample_bytree,n_estimators,test_size,best_score]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print "########################## Round Time Stamp ==== " + timestamp

        grid_search_pd.to_csv(grid_search_file, index=False)



