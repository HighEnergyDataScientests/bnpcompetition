import pandas as pd

import numpy as np
import time
import itertools
import random
import copy

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

from sklearn.svm import SVC
from sklearn import ensemble
import sklearn.metrics
from sklearn.cross_validation import train_test_split
from scipy import stats


print('Load data...')
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')

output_col_name = "target"
test_col_name = "PredictedProb"
id_col_name = "ID"


id_col_name = "t_id"
timestr = time.strftime("%Y%m%d-%H%M%S")

test_size = 0.1
num_of_trial_comb = 20
score_limit = 0.46

train.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1,inplace=True)
test.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1,inplace=True)


features = [s for s in train.columns.ravel().tolist() if s != output_col_name]

print("Data Manupilation")
cat_columns = list(train.select_dtypes(include=['object']).columns)
print("Categorical Features : " + str(cat_columns))

for f in cat_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))

print("## Filling missing data")
imp = Imputer(missing_values='NaN', strategy="mean", axis=0)
imp.fit(train[features])
train[features] = imp.transform(train[features])
test[features] = imp.transform(test[features])
        
X_pos = train[train[output_col_name] == 1]
X_neg = train[train[output_col_name] == 0]
    
X_train_pos, X_valid_pos = train_test_split(X_pos, test_size=test_size)
X_train_neg, X_valid_neg = train_test_split(X_neg, test_size=test_size)
    
X_train = pd.concat([X_train_pos,X_train_neg])
X_valid = pd.concat([X_valid_pos,X_valid_neg])

y_train = X_train[output_col_name]
y_valid = X_valid[output_col_name]

X_train = X_train[features]
X_valid = X_valid[features]
X_test = test[features]


n_estimators_list       = [500, 700, 900, 1000, 1100, 1300]
max_features            = ["auto","log2",10,25,50,60,100,None]
min_samp_splits         = [2,4,6,8]
max_depth               = [10,15,20,25,30,35]
min_samp_leafs          = [2,4,6,8]

combinations = list(itertools.product(n_estimators_list,max_features,min_samp_splits,max_depth,min_samp_leafs))
random_sample_to_run = random.sample(combinations,num_of_trial_comb)

pred_array = pd.DataFrame(index=test.index)

trail_num = 0

for t_par in random_sample_to_run:
    print("Training Settings = %s" % str(t_par))
    n_est               = t_par[0]
    max_feat            = t_par[1]
    min_samp            = t_par[2]
    max_dep             = t_par[3]
    min_samp            = t_par[4]

    trail_num = trail_num + 1
    
    print('Training... %d' % trail_num)
    extc = ExtraTreesClassifier(n_estimators=n_est,criterion='entropy',max_features=max_feat,min_samples_split=min_samp,max_depth= max_dep, min_samples_leaf= min_samp, n_jobs = -1)      

    extc.fit(X_train,y_train) 

    print('Predict... %d' % trail_num)
    p_train = extc.predict_proba(X_train)
    p_valid = extc.predict_proba(X_valid)

    score_train = sklearn.metrics.log_loss(y_train, p_train[:,1])
    score_valid = sklearn.metrics.log_loss(y_valid, p_valid[:,1])
    print("Score based on training data set = %s" % str(score_train))
    print("Score based on validating data set = %s" % str(score_valid))
    
    if score_valid < score_limit:
        pred_name = "pred_%d" % trail_num
        pred_array[pred_name] = extc.predict_proba(X_test)[:,1]
    

# Using Harmonic Mean.
test[test_col_name] = stats.hmean(pred_array,axis=1)
test[[id_col_name,test_col_name]].to_csv("../predictions/pred_extra_" + timestr + ".csv", index=False)

