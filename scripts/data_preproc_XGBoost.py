#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics team on kaggle
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################
#
############################### This is data_preproc_XGBoost.py module ############################################
import time
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import Imputer

from sklearn.cross_validation import train_test_split

# Specific controlling parameters for data_preproc_XGBoost.py module
enable_feature_analysis = 1
num_iterations = 5
save_limit = 0.455
num_of_trial_comb = 20
missing_data_corr_cut_off = 0.02
exhaustive_grid_search = 0
do_bayesian_features = 1

def analyze_data(drop_columns, train,test):
    print("## Train XGBoost models to fill missing columns............")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    
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




def data_proc_XGBoost(models_predictions_file,train,test, output_col_name,test_col_name,id_col_name,test_size_test,test_size_val):

   timestamp = time.strftime("%Y%m%d-%H%M%S")
   print("########################## Time Stamp ==== " + timestamp)
   print("## Data Processing")
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

   if do_bayesian_features == 1:
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

           lbl = preprocessing.LabelEncoder()
           lbl.fit(list(train[f].values) + list(test[f].values))
           train[f] = lbl.transform(list(train[f].values))
           test[f] = lbl.transform(list(test[f].values))


   drop_columns = ["ID","target"]
   analyze_data(drop_columns, train,test)

   train["na_count"] = train.count(axis=1)
   test["na_count"] = test.count(axis=1)

   print("## Calculating Correlation")
   correlation_p = train.corr()
   train_av = train.isnull().astype(int)
   train_av['target'] = train['target']
   train_av['ID'] = train['ID']
   test_av = test.isnull().astype(int)
   test_av["ID"] = test["ID"]
   train_av_corr = train_av.corr()
   target_corr_missing = train_av_corr["target"].fillna(0).sort_values()
   target_corr_missing_most_neg = list(target_corr_missing[target_corr_missing < -missing_data_corr_cut_off].index.ravel())
   target_corr_missing_most_pos = list(set(list(target_corr_missing[target_corr_missing >  missing_data_corr_cut_off].index.ravel())) - set(["target"]))
   added_missing_features = target_corr_missing_most_neg + target_corr_missing_most_pos
   missing_features_new_names = [t + "_available" for t in added_missing_features]
   #print missing_features_new_names
   rename_dict = dict(zip(added_missing_features, missing_features_new_names))
   #print rename_dict
   train_av.rename(columns=rename_dict,inplace=True)
   test_av.rename(columns=rename_dict,inplace=True)
   missing_features_new_names.append("ID")
   train = pd.merge(train,train_av[missing_features_new_names],on="ID")
   test = pd.merge(test,test_av[missing_features_new_names],on="ID")
   print (train.columns.ravel())
   print (test.columns.ravel())
   
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