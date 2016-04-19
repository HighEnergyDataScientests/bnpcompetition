#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics team on kaggle
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################
#
############################### This is two_layer_training module #################################################

import numpy as  np

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

#fixing random state
random_state=1

def First_layer_classifiers (X, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test):
    #Parameters and data for XGBoost clissifier
    num_boost_round = 5000
    params = {"objective": "binary:logistic",
          "eta": 0.03,
          "nthread":3,
          "max_depth": 6,
          "subsample": 0.67,
          "colsample_bytree": 0.9,
          "eval_metric": "logloss",
          "n_estimators": 100,
          "silent": 1,
          "seed": 93425
          }
          
    #Defining the classifiers
    clfs = {'LRC'  : LogisticRegression(n_jobs=-1, random_state=random_state), 
            'SVM' : SVC(probability=True, max_iter=100, random_state=random_state), 
            'RFC'  : RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                       random_state=random_state), 
            'GBM' : GradientBoostingClassifier(n_estimators=50, 
                                           random_state=random_state), 
            'ETC' : ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
                                     random_state=random_state),
            'KNN' : KNeighborsClassifier(n_neighbors=30, n_jobs=-1),
            'ABC' : AdaBoostClassifier(DecisionTreeClassifier(max_depth=30),
                                       algorithm="SAMME", n_estimators=350,
                                       random_state=random_state),
            'SGD' : SGDClassifier(loss="log", n_iter = 1100, n_jobs=-1,
                                  random_state=random_state),
            'DTC' : DecisionTreeClassifier(max_depth=7, random_state=random_state),
            'XGB' : 'XGB'
            }
            
    
    #predictions on the validation and test sets
    p_valid = []
    p_test = []
    p_ttest_t = []
    
    print('')
    print('Performance of individual classifiers (1st layer) on X_test')   
    print('-----------------------------------------------------------')
   
    for model_name, clf in clfs.items():
       #First run. Training on (X_train, y_train) and predicting on X_valid.
       if model_name == 'XGB':
           dtrain = xgb.DMatrix(X_train, y_train)
           dvalid = xgb.DMatrix(X_valid, y_valid)
           watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
           gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, verbose_eval=True)
           print("Best Score for this run : " + str(gbm.best_score))
           y_xgb = gbm.predict(xgb.DMatrix(X_valid),ntree_limit=gbm.best_ntree_limit)
           #
           one = np.ones(shape=(y_xgb.shape))
           q = np.subtract(one, y_xgb)
           yv = np.column_stack((q,y_xgb))
       else:
          clf.fit(X_train, y_train)
          yv = clf.predict_proba(X_valid)

       p_valid.append(yv)
       
       #Printing out the performance of the classifier
       print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(model_name), 'logloss_val  =>', log_loss(y_valid, yv[:,1])))
       
       #Second run. Training on (X, y) and predicting on X_test.
       if model_name == 'XGB':
           dtrain = xgb.DMatrix(X, y)
           dvalid = xgb.DMatrix(X_test, y_test)
           watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
           gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, verbose_eval=True)
           print("Best Score for this run : " + str(gbm.best_score))
           y_xgb_test = gbm.predict(xgb.DMatrix(X_test),ntree_limit=gbm.best_ntree_limit)
           y_xgb_ttest = gbm.predict(xgb.DMatrix(temp_test),ntree_limit=gbm.best_ntree_limit)
           #
           one_test = np.ones(shape=(y_xgb_test.shape))
           q_test = np.subtract(one_test, y_xgb_test)
           yt = np.column_stack((q_test,y_xgb_test))
           #
           one_ttest = np.ones(shape=(y_xgb_ttest.shape))
           q_ttest = np.subtract(one_ttest, y_xgb_ttest)
           yt_tt = np.column_stack((q_ttest,y_xgb_ttest))
       else:
          clf.fit(X, y)
          yt = clf.predict_proba(X_test)
          yt_tt = clf.predict_proba(temp_test)

       p_test.append(yt)
       p_ttest_t.append(yt_tt)
       
       #Printing out the performance of the classifier
       print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(model_name), 'logloss_test  =>', log_loss(y_test, yt[:,1])))
       print('')

    return p_valid, p_test, p_ttest_t


def Second_layer_ensembling (p_valid, p_test, y_valid, y_test, p_ttest_t):
    print('')
    print('Performance of optimization based ensemblers (2nd layer) on X_test')
    print('------------------------------------------------------------------')
    
    #Creating the data for the 2nd layer
    XV = np.hstack(p_valid)
    XT = np.hstack(p_test)

    XTTT = np.hstack(p_ttest_t)
    
    clf = LogisticRegressionCV(scoring='log_loss', random_state=random_state)
    clf = clf.fit(XV, y_valid)
    
    yT = clf.predict_proba(XT)
    yt_out = clf.predict_proba(XTTT)
    print('{:20s} {:2s} {:1.7f}'.format('Ensemble of Classifiers', 'logloss_ensembled  =>', log_loss(y_test, yT[:,1])))
    
    return yt_out