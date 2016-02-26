import pandas as pd
train = pd.read_csv('../inputs/train.csv')
train.head()
#train['v50']
#train.isnull().sum()
col_nan_count = train.isnull().sum()
#col_nan_count.head()
#col_nan_count[col_nan_count[0] == 0]
#col_nan_count[col_nan_count == 0]
#train[v24]
#train['v24']
#col_nan_count[col_nan_count == 0]
#train['v38']
#pd.unique(train['v38'])
#col_nan_count[col_nan_count == 0]
#pd.unique(train['v47'])
#pd.unique(train['v62'])
#pd.unique(train['v66'])
#pd.unique(train['v67'])
#pd.unique(train['v68'])
#pd.unique(train['v62'])
#pd.unique(train['v71'])
#pd.unique(train['v72'])
#pd.unique(train['v74'])
#pd.unique(train['v75'])
#pd.unique(train['v79'])
#pd.unique(train['v110'])
#pd.unique(train['v129'])
#import readline
#readline.write_history_file("kmeans_analysis.py")
parameters_names = col_nan_count[col_nan_count == 0].index.ravel()

print(parameters_names)
