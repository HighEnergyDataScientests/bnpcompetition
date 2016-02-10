# -----------------------------------------------------------------------------
#  Name: categorical_plots
#  Purpose: Plot bar graphs for all categorical fields
#
# -----------------------------------------------------------------------------
"""
Make a bar graph for all categorical fields.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Agg")  # Needed to save figures
import matplotlib.pyplot as plt

# Import data
print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')

# Process data
print("## Data Processing")
output_col_name = "target"
id_col_name = "ID"

train = train.drop(id_col_name, axis=1)

# train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))

# Split data into 2 dataframes, based on the target 
df_pos = train.loc[train[output_col_name] == 1]
df_neg = train.loc[train[output_col_name] == 0]

# Create list of features
features = [s for s in train.columns.ravel().tolist() if s != output_col_name]

# Create plots
print("## Creating plots")
for f in features:
    if train[f].dtype.name == 'object':
        
        # Get the counts of the categories for both pos/neg series
        counts_pos = df_pos[f].value_counts().sort_index()
        counts_neg = df_neg[f].value_counts().sort_index()
        counts_pos.name = 'pos'
        counts_neg.name = 'neg'

        # Combine into one dataframe
        feat_counts_df = pd.concat([counts_pos, counts_neg], axis=1)

        # Plot
        plt.clf() # Clear figure
        feat_counts_df.plot(kind='bar')

        # Labels
        plt.xlabel(f)
        plt.ylabel('Freq')
        plt.title(f + ' Bar Graph')
        plt.legend()

        # Save
        plt.savefig('plots_cat/' + f + '.png')

        # Do some calculations
        feat_counts_df['sum'] = feat_counts_df.sum(axis=1)
        feat_counts_df['pos_pct'] = feat_counts_df['pos']/feat_counts_df['sum']

        feat_counts_df.to_csv('plots_cat/stats/' + f + '.csv')


    else: # draw histograms
        plt.clf()
        plt.xlabel(f)
        plt.ylabel('Freq')
        plt.title(f + ' Histogram')

        plt.hist(df_pos[f].dropna(), label='pos', bins=20)
        plt.hist(df_neg[f].dropna(), label='neg', bins=20)

        plt.legend()

        plt.savefig('plots_histos/' + f + '.png')
