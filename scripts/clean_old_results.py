#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import os


### Controlling Parameters
clean_up_limit = 0.465

models_predictions_file = "../predictions/models_predictions.csv"

if os.path.isfile(models_predictions_file):
    models_predictions = pd.read_csv(models_predictions_file)
else:
    print("No model_predictions found. Ignore")
    exit()
 

# Find the predictions that are less than limit
clean_predictions = list(models_predictions.loc[0][models_predictions.loc[0] > clean_up_limit].index.ravel())

for pred in clean_predictions:
    models_predictions.drop(pred, axis=1, inplace=True)
    
    timestr = pred.replace("run_","")
    
    pred_file       = "../predictions/pred_" + timestr + ".csv"
    feat_file       = '../feature_analysis/feature_importance_xgb_' + timestr + '.png'
    feat_imp_file   = '../feature_analysis/feature_importance_xgb_' + timestr + '.csv'
    feat_map_file   = '../feature_analysis/xgb_' + timestr + '.fmap'
    
    if os.path.isfile(pred_file):
        print("File " + pred_file + " found and removed.")
        os.remove(pred_file)
    
    if os.path.isfile(feat_file):
        print("File " + feat_file + " found and removed.")
        os.remove(feat_file)    
    
    if os.path.isfile(feat_imp_file):
        print("File " + feat_imp_file + " found and removed.")
        os.remove(feat_imp_file) 
    
    if os.path.isfile(feat_map_file):
        print("File " + feat_map_file + " found and removed.")
        os.remove(feat_map_file)

models_predictions.to_csv(models_predictions_file, index=False)
