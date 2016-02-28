# -----------------------------------------------------------------------------
#  Name: correlation
#  Purpose: Calculate correlations and covariance
#
#
# -----------------------------------------------------------------------------
"""
Calculate correlations and covariance
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

# --- Import data ---
print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')

# --- Process data ---
print("## Data Processing")

# Define parameters
output_col_name = "target"
id_col_name = "ID"

train = train.drop(id_col_name, axis=1)


# --- Calculate matrices and save to csv ---
print("## Calculating matrices")

print("    - Pearson correlation matrix")
correlation_p = train.corr()  # Pearson method
correlation_p.to_csv('../feature_analysis/stats/correlation_matrix_pearson.csv')

# print("    - Kendall Tau correlation matrix")
# correlation_k = train.corr(method='kendall')    # Kendall Tau
# correlation_k.to_csv('stats/correlation_matrix_kendall.csv')

print("    - Spearman correlation matrix")
correlation_s = train.corr(method='spearman')   # Spearman
correlation_s.to_csv('../feature_analysis/stats/correlation_matrix_spearman.csv')

covariance = train.cov()
covariance.to_csv('../feature_analysis/stats/covariance_matrix.csv')

# --- Plot matrices ---
print("## Plotting")
plt.matshow(correlation_p)
plt.savefig('../feature_analysis/stats/correlation_matrix_pearson.png')
plt.clf()

# plt.matshow(correlation_k)
# plt.savefig('stats/correlation_matrix_kendall.png')
# plt.clf()

plt.matshow(correlation_s)
plt.savefig('../feature_analysis/stats/correlation_matrix_spearman.png')
plt.clf()

plt.matshow(covariance)
plt.savefig('../feature_analysis/stats/covariance_matrix.png')
plt.clf()