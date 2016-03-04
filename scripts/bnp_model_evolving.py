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
import copy

from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Mutators
from pyevolve import Crossovers
from pyevolve import Initializators
from pyevolve import GAllele
from pyevolve import Consts


def ceate_feature_map(features,featureMapFile):
    outfile = open(featureMapFile, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def eval_func(chromosome):
    t_par = chromosome.getInternalList()
    print("## Start with Individual : " + str(t_par))
        
    eta                 = t_par[0]
    max_depth           = t_par[1]
    subsample           = t_par[2]
    colsample_bytree    = t_par[3]
    n_estimators        = t_par[4]
    test_size           = t_par[5]
    imp_start           = t_par[6]
    num_of_feat_corr    = t_par[7]


    print("## Filling missing data")
    imp = Imputer(missing_values='NaN', strategy=imp_start, axis=0)
    imp.fit(train[features])
    train[features] = imp.transform(train[features])
    test[features] = imp.transform(test[features])

    curr_features = copy.deepcopy(features)

    print("## Creating Random features based on Correlation")
    output_cor = correlation_p[output_col_name].sort_values()

    most_neg_cor = list(output_cor.index[0:num_of_feat_corr].ravel())
    most_pos_cor = list(output_cor.index[(-2-num_of_feat_corr):-2].ravel())

    for f1, f2 in pairwise(most_neg_cor):
        train[f1 + "_" + f2] = train[f1] + train[f2]
        test[f1 + "_" + f2] = test[f1] + test[f2]
        curr_features += [f1 + "_" + f2]

    for f1, f2 in pairwise(most_pos_cor):
        train[f1 + "_" + f2] = train[f1] + train[f2]
        test[f1 + "_" + f2] = test[f1] + test[f2]
        curr_features += [f1 + "_" + f2]


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
    best_score = train_model(curr_features,params,num_boost_round,test_size)
    grid_search_pd.loc[len(grid_search_pd),grid_search_columns] = [eta,max_depth,subsample,colsample_bytree,n_estimators,test_size,imp_start,num_of_feat_corr,best_score]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Round Time Stamp ==== " + timestamp)

    grid_search_pd.to_csv(grid_search_file, index=False)

    return best_score

def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return itertools.izip(a, a)
    
def train_model(features,params,num_boost_round,test_size):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print "########################## Time Stamp ==== " + timestamp
    
    print("## Train a XGBoost model")
    print("Features: ", features)
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



if __name__ == "__main__":
    ## Finding columns with first element is less than value
    ## df.loc[0][df.loc[0] < 0.46]

    ### Controlling Parameters
    output_col_name = "target"
    test_col_name = "PredictedProb"
    enable_feature_analysis = 1
    id_col_name = "ID"
    num_iterations = 5
    termination_score = 0.43
    save_limit = 0.45
    num_of_trial_comb = 20
    exhaustive_grid_search = 0


    ### Creating output folders
    if not os.path.isdir("../predictions"):
        os.mkdir("../predictions")

    if not os.path.isdir("../intermediate_data"):
        os.mkdir("../intermediate_data")

    if not os.path.isdir("../saved_states"):
        os.mkdir("../saved_states")



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

        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


    drop_columns = ["ID","target"]
    analyze_data(drop_columns)

    train["na_count"] = train.count(axis=1)
    test["na_count"] = test.count(axis=1)

    features = [s for s in train.columns.ravel().tolist() if s != output_col_name]

    print("## Calculating Correlation")
    corr_features = features + [output_col_name]
    correlation_p = train[corr_features].corr()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("########################## Time Stamp ==== " + timestamp)
    print("## Training")
    grid_search_file = "../predictions/grid_search_file.csv"
    grid_search_columns = ['eta','max_depth','subsample','colsample_bytree','n_estimators','test_size','imputer_strategy','num_of_feat_corr','result']

    if os.path.isfile(grid_search_file):
        grid_search_pd = pd.read_csv(grid_search_file)
        grid_search_missing = list(set(grid_search_columns) - set(grid_search_pd.columns.ravel()))
        print grid_search_missing
        if len(grid_search_missing) > 0:
            grid_search_pd = pd.concat([grid_search_pd,pd.DataFrame(columns=grid_search_missing)])
            grid_search_pd['imputer_strategy'].fillna('mean',inplace=True)
            grid_search_pd['num_of_feat_corr'].fillna(0,inplace=True)
    else:
        grid_search_pd = pd.DataFrame(columns=grid_search_columns)
    
    ######################################################
    ### Start Evolutionary Algorithm
    ######################################################
    ## Genome Alleles instance
    setOfAlleles = GAllele.GAlleles()
    
    # eta
    a = GAllele.GAlleleRange(0.005, 0.4,real=True)
    setOfAlleles.add(a)
    
    # max_depth_list
    a = GAllele.GAlleleRange(3, 20,real=False)
    setOfAlleles.add(a)
    
    # subsample_list
    a = GAllele.GAlleleRange(0.45, 0.99,real=True)
    setOfAlleles.add(a)
    
    # colsample_bytree_list
    a = GAllele.GAlleleRange(0.45, 0.99,real=True)
    setOfAlleles.add(a)
    
    # n_estimators_list
    a = GAllele.GAlleleRange(5, 200,real=False)
    setOfAlleles.add(a)
    
    # test_size_list
    a = GAllele.GAlleleRange(0.048, 0.052,real=True)
    setOfAlleles.add(a)
    
    # imputer_st_list
    a = GAllele.GAlleleList(['mean',"median"])
    setOfAlleles.add(a)
    
    # num_of_feat_corr_list
    a = GAllele.GAlleleRange(2, 20,real=False)
    setOfAlleles.add(a)

    ## Creating List of Alleles
    genome = G1DList.G1DList(8)
    genome.setParams(allele=setOfAlleles,bestrawscore=termination_score,rounddecimal=4)

    # The evaluator function (objective function)
    genome.evaluator.set(eval_func)

    # This mutator and initializator will take care of
    # initializing valid individuals based on the allele set
    # that we have defined before
    genome.mutator.set(Mutators.G1DListMutatorAllele)
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
    genome.crossover.set(Crossovers.G1DListCrossoverUniform)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMinimax(Consts.minimaxType["minimize"])
    ga.setGenerations(100)
    ga.setPopulationSize(20)
    ga.setMutationRate(0.5)
    ga.setElitismReplacement(4)
    
    ga.terminationCriteria.set(GSimpleGA.RawScoreCriteria)

    # Do the evolution
    ga.evolve(freq_stats=2)

    # Best individual
    print ga.bestIndividual()




