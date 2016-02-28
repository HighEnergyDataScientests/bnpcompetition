# -----------------------------------------------------------------------------
#  Name: plot_correlation
#  Purpose: Replot correlations and covariance
#
# -----------------------------------------------------------------------------
"""
Plot correlations and covariance
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
import seaborn as sns

# --- Import data ---
print("## Loading Data")
corr_p = pd.io.parsers.read_csv('../feature_analysis/stats/correlation_matrix_pearson.csv', index_col=0)
corr_k = pd.io.parsers.read_csv('../feature_analysis/stats/correlation_matrix_kendall.csv', index_col=0)
corr_s = pd.io.parsers.read_csv('../feature_analysis/stats/correlation_matrix_spearman.csv', index_col=0)
covar = pd.io.parsers.read_csv('../feature_analysis/stats/covariance_matrix.csv', index_col=0)

# --- Plot matrices ---
print("## Plotting")
sns.set_palette("coolwarm")
sns.set_style('ticks')

plot = sns.heatmap(corr_p, xticklabels=5, yticklabels=5, square=True)
plt.savefig('../feature_analysis/stats/correlation_matrix_pearson_sb.png')
plt.clf()

plot = sns.heatmap(corr_k, xticklabels=5, yticklabels=5, square=True)
plt.savefig('../feature_analysis/stats/correlation_matrix_kendall_sb.png')
plt.clf()

plot = sns.heatmap(corr_s, xticklabels=5, yticklabels=5, square=True)
plt.savefig('../feature_analysis/stats/correlation_matrix_spearman_sb.png')
plt.clf()

plot = sns.heatmap(covar, xticklabels=5, yticklabels=5, square=True)
plt.savefig('../feature_analysis/stats/covariance_matrix_sb.png')
plt.clf()

