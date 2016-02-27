# -----------------------------------------------------------------------------
#  Name: pairwise_feature_plots
#  Purpose: Plot all fields
#
# -----------------------------------------------------------------------------
"""
Create scatterplots between pairs of features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import itertools as it
import seaborn as sb

sb.set()

# --- Import data ---
print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')

# --- Process data ---
print("## Data Processing")

# Define parameters
output_col_name = "target"
id_col_name = "ID"

train = train.drop(id_col_name, axis=1)

# Split data into 2 dataframes, based on the target 
df_pos = train.loc[train[output_col_name] == 1]
df_neg = train.loc[train[output_col_name] == 0]

# --- Create list of features ---
features = [s for s in train.columns.ravel().tolist() if s != output_col_name]

# Split into categorical and numerical features
cat_feat = [f for f in features if train[f].dtype.name == 'object']
num_feat = list(set(features)-set(cat_feat))

# --- Create plots ---
print("## Creating plots")

# num_df = train.ix[:, num_feat]
# fig = sb.pairplot(num_df, hue=output_col_name)
# fig.savefig('plots_pairwise/matrix.png')

# Try for a subset of train:
train_subset = train_test_split(train, test_size=0.1)

for cat_f in cat_feat:
    for num_f in num_feat:
        swm_plt = sb.swarmplot(x=cat_f, y=num_f, hue=output_col_name, data=train_subset)
        swm_plt.savefig('../feature_analysis/plots_pairwise/' + cat_f + '_' + num_f + '.png')

