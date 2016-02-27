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
from sklearn.preprocessing import scale

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
id_col_name = "ID"
num_imp_features = 1


timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Start Time Stamp ==== " + timestamp)
print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')
timestamp = time.strftime("%Y%m%d-%H%M%S")
print("########################## Time Stamp ==== " + timestamp)
print("## Data Processing")
train = train.drop(id_col_name, axis=1)
print("## Data Encoding")

cat_columns = list(train.select_dtypes(include=['object']).columns)
print("Categorical Features : " + str(cat_columns))

for f in cat_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[f].values))
    train[f] = lbl.transform(list(train[f].values))


train["na_count"] = train.count(axis=1)

features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
print("Features: ", features)

print("## Calculating Correlation")
corr_features = features + [output_col_name]
correlation_p = train[corr_features].corr()

train.fillna(-10,inplace=True)

train[features] = scale(train[features])



print("## Creating Random features based on Correlation")
output_cor = correlation_p[output_col_name].sort_values()
output_cor = correlation_p[output_col_name].sort_values()

most_neg_cor = list(output_cor.index[0:num_imp_features].ravel())
most_pos_cor = list(output_cor.index[-num_imp_features-2:-2].ravel())

most_pos_cor_plot = most_pos_cor + [output_col_name]
most_neg_cor_plot = most_neg_cor + [output_col_name]


print("## Showing Plots")
train[most_pos_cor_plot].plot()

output_file("legend.html", title="legend.py example")

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

p1 = figure(title="Legend Example", tools=TOOLS)

p1.circle(x, y, legend="sin(x)")
p1.circle(x, 2*y, legend="2*sin(x)", color="orange", )
p1.circle(x, 3*y, legend="3*sin(x)", color="green", )

p2 = figure(title="Another Legend Example", tools=TOOLS)

p2.circle(x, y, legend="sin(x)")
p2.line(x, y, legend="sin(x)")

p2.line(x, 2*y, legend="2*sin(x)",
    line_dash=[4, 4], line_color="orange", line_width=2)

p2.square(x, 3*y, legend="3*sin(x)", fill_color=None, line_color="green")
p2.line(x, 3*y, legend="3*sin(x)", line_color="green")

show(vplot(p1, p2))  # open a browser

# Labels
#plt.xlabel(f)
#plt.ylabel('Freq')
#plt.title(f + ' Bar Graph')
plt.legend()

# Save
plt.savefig('features_trend.png')


